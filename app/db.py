"""Single embedded operational store: relational rows + FTS5 (BM25) + vectors.

One SQLite file means agent state, memory, the document tree and the vectors commit in the
same transaction, so the index never points at a deleted or half-ingested paper.

Vectors are float32 BLOBs. Search keeps an in-memory matrix of all pointer vectors (rebuilt
lazily after any write), so a query is one matrix-vector product: ~50k pointers x 768 dims
is ~150 MB and a few milliseconds on an M4. Swap `VectorCache` for sqlite-vec or FAISS if
a library grows past that.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA_VERSION = 2
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, doi TEXT, arxiv_id TEXT,
  filename TEXT, file_path TEXT, status TEXT DEFAULT 'queued',   -- queued | processing | ready | failed
  progress TEXT DEFAULT '', error TEXT DEFAULT '', parser TEXT DEFAULT '',
  page_count INTEGER, node_count INTEGER, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS nodes (           -- skeleton tree: whole sections, tables, figures
  node_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers ON DELETE CASCADE,
  parent_id TEXT, kind TEXT, level INTEGER, ord INTEGER, title TEXT, breadcrumb TEXT,
  text TEXT, token_count INTEGER, page_start INTEGER, page_end INTEGER, embeddable INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS nodes_paper ON nodes(paper_id);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);
CREATE TABLE IF NOT EXISTS pointers (        -- small embedded proxies that point at nodes
  pointer_id INTEGER PRIMARY KEY, node_id TEXT REFERENCES nodes ON DELETE CASCADE,
  paper_id TEXT REFERENCES papers ON DELETE CASCADE, text TEXT, embedding BLOB, embed_model TEXT, dim INTEGER
);
CREATE INDEX IF NOT EXISTS pointers_paper ON pointers(paper_id);
CREATE VIRTUAL TABLE IF NOT EXISTS pointers_fts USING fts5(text, content='pointers', content_rowid='pointer_id');
CREATE TRIGGER IF NOT EXISTS pointers_ad AFTER DELETE ON pointers BEGIN
  INSERT INTO pointers_fts(pointers_fts, rowid, text) VALUES ('delete', old.pointer_id, old.text);
END;

CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT DEFAULT '', created_at REAL,
                                     updated_at REAL, summary TEXT DEFAULT '', summarized_upto INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, run_id TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id);
CREATE TABLE IF NOT EXISTS runs (            -- checkpoint after every node -> resumable, replayable
  run_id TEXT PRIMARY KEY, session_id TEXT, state_json TEXT, status TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT, seq INTEGER, type TEXT, data TEXT, ts REAL, PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS episodes (        -- episodic memory: what happened
  id INTEGER PRIMARY KEY, session_id TEXT, run_id TEXT, summary TEXT, embedding BLOB, created_at REAL
);
CREATE TABLE IF NOT EXISTS facts (           -- semantic memory: distilled, invalidatable
  id INTEGER PRIMARY KEY, text TEXT, kind TEXT, source_ref TEXT, embedding BLOB,
  valid INTEGER DEFAULT 1, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS notes (           -- research notebook written by the agent
  id INTEGER PRIMARY KEY, session_id TEXT, claim TEXT, evidence TEXT, node_id TEXT, created_at REAL
);
"""


def to_blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


# Function words carry no topic, but BM25 still rewards them; drop them from keyword queries.
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how
if in into is it its may might of on or our paper papers should so than that the their them then there these
they this those to us use used uses using was we were what when where which who why will with would you your
about across all also any between both each more most other over such through under very""".split())


@dataclass
class Hit:
    node_id: str
    paper_id: str
    breadcrumb: str
    score: float


class VectorCache:
    """All pointer vectors as one normalised matrix; rebuilt lazily after writes."""

    def __init__(self) -> None:
        self.ids = np.zeros(0, dtype=np.int64)
        self.papers: list[str] = []
        self.mat = np.zeros((0, 0), dtype=np.float32)
        self.dirty = True

    def rebuild(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT pointer_id, paper_id, embedding FROM pointers WHERE embedding IS NOT NULL").fetchall()
        if rows:
            mat = np.vstack([from_blob(r[2]) for r in rows])
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self.ids = np.array([r[0] for r in rows], dtype=np.int64)
            self.papers = [r[1] for r in rows]
            self.mat = mat
        else:
            self.ids, self.papers, self.mat = np.zeros(0, dtype=np.int64), [], np.zeros((0, 0), dtype=np.float32)
        self.dirty = False

    def query(self, qvec: list[float], k: int, paper_ids: set[str] | None) -> list[int]:
        if not len(self.ids):
            return []
        q = np.asarray(qvec, dtype=np.float32)
        if q.shape[0] != self.mat.shape[1]:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.mat.shape[1]}: "
                             "the embedding model changed; re-ingest or run `python -m app reindex`")
        sims = self.mat @ (q / (np.linalg.norm(q) + 1e-9))
        if paper_ids:
            mask = np.array([p in paper_ids for p in self.papers])
            sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [int(self.ids[i]) for i in order if np.isfinite(sims[i])]


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.conn.commit()
        self._lock = threading.RLock()  # serialises writers from worker threads
        self.vectors = VectorCache()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.conn:  # commit on success, rollback on exception
            yield self.conn

    def q(self, sql: str, args: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    def q1(self, sql: str, args: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, args).fetchone()

    # ---------- papers ----------
    def upsert_paper_stub(self, paper_id: str, filename: str, file_path: str) -> str:
        """Register an upload. Returns the paper's current status (dedupe by content hash)."""
        row = self.q1("SELECT status FROM papers WHERE paper_id=?", (paper_id,))
        if row and row["status"] in ("ready", "queued", "processing"):
            return row["status"]
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO papers(paper_id, title, filename, file_path, status, progress, error, "
                      "created_at, updated_at) VALUES (?,?,?,?, 'queued', '', '', ?, ?)",
                      (paper_id, Path(filename).stem, filename, file_path, now, now))
        return "queued"

    def set_paper_status(self, paper_id: str, status: str, progress: str = "", error: str = "") -> None:
        with self.tx() as c:
            c.execute("UPDATE papers SET status=?, progress=?, error=?, updated_at=? WHERE paper_id=?",
                      (status, progress, error, time.time(), paper_id))

    def papers(self) -> list[dict]:
        rows = self.q("SELECT paper_id, title, authors, year, doi, arxiv_id, filename, status, progress, error, "
                      "page_count, node_count, created_at FROM papers ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def paper(self, paper_id: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM papers WHERE paper_id=?", (paper_id,))

    def pending_papers(self) -> list[sqlite3.Row]:
        return self.q("SELECT paper_id, file_path FROM papers WHERE status IN ('queued','processing') "
                      "ORDER BY created_at")

    def delete_paper(self, paper_id: str) -> str | None:
        row = self.paper(paper_id)
        if row is None:
            return None
        with self.tx() as c:
            c.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))  # cascades to nodes and pointers
        self.vectors.dirty = True
        return row["file_path"]

    def clear_paper_index(self, paper_id: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM pointers WHERE paper_id=?", (paper_id,))
            c.execute("DELETE FROM nodes WHERE paper_id=?", (paper_id,))
        self.vectors.dirty = True

    def node(self, node_id: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM nodes WHERE node_id=?", (node_id,))

    def children(self, node_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM nodes WHERE parent_id=? ORDER BY ord", (node_id,))

    def paper_tree(self, paper_id: str) -> list[dict]:
        rows = self.q("SELECT node_id, parent_id, kind, level, ord, title, breadcrumb, token_count, page_start, "
                      "page_end FROM nodes WHERE paper_id=? ORDER BY level, ord", (paper_id,))
        return [dict(r) for r in rows]

    # ---------- hybrid retrieval: BM25 + vector, fused with Reciprocal Rank Fusion ----------
    def search(self, query: str, qvec: list[float] | None, k: int = 8,
               paper_ids: list[str] | None = None, rrf_k: int = 60) -> list[Hit]:
        scope = set(paper_ids) if paper_ids else None
        flt, args = "", []
        if scope:
            flt = f" AND p.paper_id IN ({','.join('?' * len(scope))})"
            args = list(scope)

        ranks: dict[int, float] = {}
        # 1) keyword (BM25). Quote tokens so user punctuation can't break FTS syntax.
        terms = [t for t in "".join(ch if ch.isalnum() else " " for ch in query).split()
                 if len(t) > 2 and t.lower() not in STOPWORDS]
        if terms:
            fts_q = " OR ".join(f'"{t}"' for t in terms)
            rows = self.q(f"SELECT p.pointer_id FROM pointers_fts f JOIN pointers p ON p.pointer_id=f.rowid "
                          f"WHERE pointers_fts MATCH ?{flt} ORDER BY bm25(pointers_fts) LIMIT 50", [fts_q, *args])
            for r, row in enumerate(rows):
                ranks[row[0]] = ranks.get(row[0], 0) + 1 / (rrf_k + r + 1)
        # 2) semantic (cosine) over the cached matrix
        if qvec is not None:
            with self._lock:
                if self.vectors.dirty:
                    self.vectors.rebuild(self.conn)
            for r, pid in enumerate(self.vectors.query(qvec, 50, scope)):
                ranks[pid] = ranks.get(pid, 0) + 1 / (rrf_k + r + 1)

        best: dict[str, Hit] = {}  # collapse multiple pointers to the same node
        for pid, score in sorted(ranks.items(), key=lambda x: -x[1]):
            row = self.q1("SELECT n.node_id, n.paper_id, n.breadcrumb FROM pointers p JOIN nodes n USING(node_id) "
                          "WHERE p.pointer_id=?", (pid,))
            if row and row["node_id"] not in best:
                best[row["node_id"]] = Hit(row["node_id"], row["paper_id"], row["breadcrumb"], score)
            if len(best) >= k:
                break
        return list(best.values())

    # ---------- sessions, runs, events ----------
    def ensure_session(self, session_id: str, title: str = "") -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at) VALUES (?,?,?,?)",
                      (session_id, title, now, now))
            if title:
                c.execute("UPDATE sessions SET title=? WHERE session_id=? AND title=''", (title, session_id))

    def sessions(self, limit: int = 50) -> list[dict]:
        rows = self.q("SELECT session_id, title, created_at, updated_at FROM sessions "
                      "ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def add_message(self, session_id: str, role: str, content: str, run_id: str | None = None) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT INTO messages(session_id, role, content, run_id, created_at) VALUES (?,?,?,?,?)",
                      (session_id, role, content, run_id, now))
            c.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))

    def messages(self, session_id: str) -> list[dict]:
        rows = self.q("SELECT role, content, run_id, created_at FROM messages WHERE session_id=? ORDER BY id",
                      (session_id,))
        return [dict(r) for r in rows]

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.q("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                      (session_id, limit))
        return [dict(r) for r in reversed(rows)]

    def session_summary(self, session_id: str) -> str:
        row = self.q1("SELECT summary FROM sessions WHERE session_id=?", (session_id,))
        return row["summary"] if row else ""

    def checkpoint(self, run_id: str, session_id: str, state_json: str, status: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
                      "state_json=excluded.state_json, status=excluded.status, updated_at=excluded.updated_at",
                      (run_id, session_id, state_json, status, time.time()))

    def load_run(self, run_id: str) -> str | None:
        row = self.q1("SELECT state_json FROM runs WHERE run_id=?", (run_id,))
        return row["state_json"] if row else None

    def interrupted_runs(self) -> list[str]:
        return [r["run_id"] for r in self.q("SELECT run_id FROM runs WHERE status='running'")]

    def add_event(self, run_id: str, seq: int, type_: str, data: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)",
                      (run_id, seq, type_, json.dumps(data), time.time()))

    def events_since(self, run_id: str, after_seq: int = -1) -> list[dict]:
        rows = self.q("SELECT seq, type, data, ts FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                      (run_id, after_seq))
        return [{"run_id": run_id, "seq": r["seq"], "type": r["type"], "data": json.loads(r["data"]), "ts": r["ts"]}
                for r in rows]

    def last_seq(self, run_id: str) -> int:
        row = self.q1("SELECT MAX(seq) AS s FROM events WHERE run_id=?", (run_id,))
        return row["s"] if row and row["s"] is not None else -1

    # ---------- memory ----------
    def add_note(self, session_id: str, claim: str, evidence: str, node_id: str) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO notes(session_id, claim, evidence, node_id, created_at) VALUES (?,?,?,?,?)",
                            (session_id, claim, evidence, node_id, time.time()))
            return int(cur.lastrowid)

    def notes(self, session_id: str, limit: int = 10) -> list[dict]:
        rows = self.q("SELECT claim, node_id FROM notes WHERE session_id=? ORDER BY id DESC LIMIT ?",
                      (session_id, limit))
        return [dict(r) for r in rows]

    def facts(self) -> list[dict]:
        rows = self.q("SELECT id, text, kind, source_ref, created_at FROM facts WHERE valid=1 ORDER BY id DESC")
        return [dict(r) for r in rows]

    def top_facts(self, qvec: list[float] | None, k: int = 5) -> list[str]:
        rows = self.q("SELECT text, embedding FROM facts WHERE valid=1")
        if not rows:
            return []
        if qvec is None:
            return [r["text"] for r in rows[-k:]]
        q = np.asarray(qvec, dtype=np.float32)
        scored = []
        for r in rows:
            v = from_blob(r["embedding"]) if r["embedding"] else None
            ok = v is not None and v.shape == q.shape
            s = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q) + 1e-9)) if ok else 0.0
            scored.append((s, r["text"]))
        return [t for _, t in sorted(scored, reverse=True)[:k]]
