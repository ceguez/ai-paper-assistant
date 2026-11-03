"""A database created by an older schema must be upgraded in place, not fail with "no such column"."""
import sqlite3

from app.db import SCHEMA_VERSION, Store

# Tables as the be-skeleton-prod-prototype branch created them (schema v1, no meta table).
V1 = """
CREATE TABLE papers (paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, doi TEXT,
  arxiv_id TEXT, file_path TEXT, status TEXT DEFAULT 'pending', created_at REAL);
CREATE TABLE nodes (node_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers ON DELETE CASCADE,
  parent_id TEXT, kind TEXT, level INTEGER, ord INTEGER, title TEXT, breadcrumb TEXT,
  text TEXT, token_count INTEGER, page_start INTEGER, page_end INTEGER, embeddable INTEGER DEFAULT 1);
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at REAL, summary TEXT DEFAULT '');
INSERT INTO papers VALUES ('p1', 'Old paper', '', 2017, '', '', '/x/Old Paper.pdf', 'ready', 100.0);
INSERT INTO papers VALUES ('p2', 'Waiting', '', 2020, '', '', '/x/w.pdf', 'pending', 200.0);
INSERT INTO nodes VALUES ('n1','p1',NULL,'section',1,0,'Intro','Intro','t',5,1,3,1);
INSERT INTO nodes VALUES ('n2','p1','n1','section',2,0,'More','Intro > More','t',5,4,9,1);
INSERT INTO sessions VALUES ('s1', 50.0, '');
"""


def test_old_database_is_migrated_in_place(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(V1)
    conn.close()

    store = Store(path)

    papers = {p["paper_id"]: p for p in store.papers()}  # used to raise: no such column: filename
    assert papers["p1"]["filename"] == "Old Paper.pdf"
    assert papers["p1"]["node_count"] == 2 and papers["p1"]["page_count"] == 9
    assert papers["p1"]["progress"] == "" and papers["p1"]["status"] == "ready"
    assert [r["paper_id"] for r in store.pending_papers()] == ["p2"]  # 'pending' -> 'queued'

    assert store.sessions()[0]["title"] == ""  # used to raise: no such column: title
    store.ensure_session("s1", "renamed")
    store.add_message("s1", "user", "hi")
    assert store.q1("SELECT summarized_upto FROM sessions WHERE session_id='s1'")[0] == 0
    assert store.q1("SELECT value FROM meta WHERE key='schema_version'")[0] == str(SCHEMA_VERSION)

    Store(path)  # second start is a no-op


def test_fresh_database_needs_no_migration(tmp_path):
    store = Store(str(tmp_path / "new.db"))
    assert store.papers() == [] and store.sessions() == []
