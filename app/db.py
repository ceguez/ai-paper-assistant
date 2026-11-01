"""Single embedded operational store: relational rows + FTS5 (BM25) + vectors.

Why one SQLite file: agent state, memory, the document tree and the vectors commit in the
same transaction, so the index can never point at a deleted or half-ingested paper.
Vectors are stored as float32 BLOBs and scored with numpy (fine to ~10^5-10^6 pointers);
swap `VectorIndex` for sqlite-vec / pgvector / FAISS without touching callers.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, doi TEXT, arxiv_id TEXT,
  file_path TEXT, status TEXT DEFAULT 'pending', created_at REAL
);
CREATE TABLE IF NOT EXISTS nodes (           -- skeleton tree: whole sections, tables, figures
  node_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers ON DELETE CASCADE,
  parent_id TEXT, kind TEXT, level INTEGER, ord INTEGER, title TEXT, breadcrumb TEXT,
  text TEXT, token_count INTEGER, page_start INTEGER, page_end INTEGER, embeddable INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS pointers (        -- small embedded proxies that point at nodes
  pointer_id INTEGER PRIMARY KEY, node_id TEXT REFERENCES nodes ON DELETE CASCADE,
  paper_id TEXT, text TEXT, embedding BLOB, embed_model TEXT, dim INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS pointers_fts USING fts5(text, content='pointers', content_rowid='pointer_id');

CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, created_at REAL, summary TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, run_id TEXT, created_at REAL
);
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


@dataclass
class Hit:
    node_id: str
    paper_id: str
    breadcrumb: str
    score: float


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:  # commit on success, rollback on exception
            yield self.conn

    # ---------- documents ----------
    def paper_exists(self, paper_id: str) -> bool:
        row = self.conn.execute("SELECT status FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return bool(row and row["status"] == "ready")

    def node(self, node_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()

    def children(self, node_id: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM nodes WHERE parent_id=? ORDER BY ord", (node_id,)).fetchall()

    # ---------- hybrid retrieval: BM25 + vector, fused with Reciprocal Rank Fusion ----------
    def search(self, query: str, qvec: list[float] | None, k: int = 8,
               paper_ids: list[str] | None = None, rrf_k: int = 60) -> list[Hit]:
        flt, args = "", []
        if paper_ids:
            flt = f" AND p.paper_id IN ({','.join('?' * len(paper_ids))})"
            args = list(paper_ids)

        ranks: dict[int, float] = {}
        # 1) keyword (BM25). Quote tokens so user punctuation can't break FTS syntax.
        fts_q = " OR ".join(f'"{t}"' for t in query.replace('"', " ").split() if len(t) > 2)
        if fts_q:
            rows = self.conn.execute(
                f"SELECT p.pointer_id FROM pointers_fts f JOIN pointers p ON p.pointer_id=f.rowid "
                f"WHERE pointers_fts MATCH ?{flt} ORDER BY bm25(pointers_fts) LIMIT 50", [fts_q, *args]).fetchall()
            for r, row in enumerate(rows):
                ranks[row[0]] = ranks.get(row[0], 0) + 1 / (rrf_k + r + 1)
        # 2) semantic (cosine). Brute force over float32 blobs.
        if qvec is not None:
            rows = self.conn.execute(
                f"SELECT p.pointer_id, p.embedding FROM pointers p WHERE p.embedding IS NOT NULL{flt}", args).fetchall()
            if rows:
                ids = [r[0] for r in rows]
                mat = np.vstack([from_blob(r[1]) for r in rows])
                q = np.asarray(qvec, dtype=np.float32)
                sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
                for r, idx in enumerate(np.argsort(-sims)[:50]):
                    pid = ids[int(idx)]
                    ranks[pid] = ranks.get(pid, 0) + 1 / (rrf_k + r + 1)

        best: dict[str, Hit] = {}  # collapse multiple pointers to the same node
        for pid, score in sorted(ranks.items(), key=lambda x: -x[1]):
            row = self.conn.execute(
                "SELECT n.node_id, n.paper_id, n.breadcrumb FROM pointers p JOIN nodes n USING(node_id) "
                "WHERE p.pointer_id=?", (pid,)).fetchone()
            if row and row["node_id"] not in best:
                best[row["node_id"]] = Hit(row["node_id"], row["paper_id"], row["breadcrumb"], score)
            if len(best) >= k:
                break
        return list(best.values())

    # ---------- sessions, runs, events ----------
    def ensure_session(self, session_id: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?,?)", (session_id, time.time()))

    def add_message(self, session_id: str, role: str, content: str, run_id: str | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO messages(session_id, role, content, run_id, created_at) VALUES (?,?,?,?,?)",
                      (session_id, role, content, run_id, time.time()))

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                                 (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def session_summary(self, session_id: str) -> str:
        row = self.conn.execute("SELECT summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return row["summary"] if row else ""

    def checkpoint(self, run_id: str, session_id: str, state_json: str, status: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
                      "state_json=excluded.state_json, status=excluded.status, updated_at=excluded.updated_at",
                      (run_id, session_id, state_json, status, time.time()))

    def load_run(self, run_id: str) -> str | None:
        row = self.conn.execute("SELECT state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return row["state_json"] if row else None

    def add_event(self, run_id: str, seq: int, type_: str, data: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO events VALUES (?,?,?,?,?)", (run_id, seq, type_, json.dumps(data), time.time()))

    def events_since(self, run_id: str, after_seq: int = -1) -> list[dict]:
        rows = self.conn.execute("SELECT seq, type, data, ts FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                                 (run_id, after_seq)).fetchall()
        return [{"run_id": run_id, "seq": r["seq"], "type": r["type"], "data": json.loads(r["data"]), "ts": r["ts"]}
                for r in rows]

    # ---------- memory ----------
    def add_note(self, session_id: str, claim: str, evidence: str, node_id: str) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO notes(session_id, claim, evidence, node_id, created_at) VALUES (?,?,?,?,?)",
                            (session_id, claim, evidence, node_id, time.time()))
            return int(cur.lastrowid)

    def notes(self, session_id: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute("SELECT claim, node_id FROM notes WHERE session_id=? ORDER BY id DESC LIMIT ?",
                                 (session_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def top_facts(self, qvec: list[float] | None, k: int = 5) -> list[str]:
        rows = self.conn.execute("SELECT text, embedding FROM facts WHERE valid=1").fetchall()
        if not rows:
            return []
        if qvec is None:
            return [r["text"] for r in rows[-k:]]
        q = np.asarray(qvec, dtype=np.float32)
        scored = []
        for r in rows:
            v = from_blob(r["embedding"]) if r["embedding"] else None
            s = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q) + 1e-9)) if v is not None else 0.0
            scored.append((s, r["text"]))
        return [t for _, t in sorted(scored, reverse=True)[:k]]
