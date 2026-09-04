"""SQLite 索引：照片与人脸 embedding 持久化。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    file_hash TEXT,
    mtime REAL,
    scanned_at TEXT NOT NULL,
    face_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    bbox TEXT NOT NULL,
    embedding BLOB NOT NULL,
    det_score REAL,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS person_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    person_name TEXT NOT NULL,
    similarity REAL,
    tagged_at TEXT NOT NULL,
    UNIQUE(photo_id, person_name),
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_tags_person ON person_tags(person_name);
"""


class Indexer:
    """管理照片与人脸索引的 SQLite 访问层。"""

    def __init__(self, db_file: Path) -> None:
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_photo_by_path(self, path: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM photos WHERE path = ?", (path,)).fetchone()
        return row

    def upsert_photo(
        self,
        path: str,
        file_hash: str | None,
        mtime: float | None,
        face_count: int,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO photos (path, file_hash, mtime, scanned_at, face_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    mtime = excluded.mtime,
                    scanned_at = excluded.scanned_at,
                    face_count = excluded.face_count
                """,
                (path, file_hash, mtime, now, face_count),
            )
            row = conn.execute("SELECT id FROM photos WHERE path = ?", (path,)).fetchone()
        assert row is not None
        return int(row["id"])

    def replace_faces(self, photo_id: int, faces: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            for face in faces:
                embedding: np.ndarray = face["embedding"]
                conn.execute(
                    """
                    INSERT INTO faces (photo_id, bbox, embedding, det_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        photo_id,
                        json.dumps(face["bbox"]),
                        embedding.astype(np.float32).tobytes(),
                        face.get("det_score"),
                    ),
                )

    def iter_all_faces(self) -> Iterator[tuple[int, int, np.ndarray, str]]:
        """Yield (face_id, photo_id, embedding, photo_path)."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT f.id AS face_id, f.photo_id, f.embedding, p.path
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
                """
            ).fetchall()
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            yield int(row["face_id"]), int(row["photo_id"]), emb, str(row["path"])

    def stats(self) -> dict[str, int]:
        with self.connection() as conn:
            photo_count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            tag_count = conn.execute("SELECT COUNT(*) FROM person_tags").fetchone()[0]
        return {
            "photos": int(photo_count),
            "faces": int(face_count),
            "tags": int(tag_count),
        }

    def add_tags(self, matches: list[dict[str, Any]]) -> int:
        """批量写入人物标签，返回新增数量。"""
        now = datetime.now(timezone.utc).isoformat()
        added = 0
        with self.connection() as conn:
            for m in matches:
                cur = conn.execute(
                    """
                    INSERT INTO person_tags (photo_id, person_name, similarity, tagged_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(photo_id, person_name) DO UPDATE SET
                        similarity = excluded.similarity,
                        tagged_at = excluded.tagged_at
                    """,
                    (m["photo_id"], m["person_name"], m["similarity"], now),
                )
                if cur.rowcount:
                    added += 1
        return added

    def get_tagged_photos(self, person_name: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.path, t.similarity
                FROM person_tags t
                JOIN photos p ON p.id = t.photo_id
                WHERE t.person_name = ?
                ORDER BY t.similarity DESC
                """,
                (person_name,),
            ).fetchall()
        return [{"path": r["path"], "similarity": r["similarity"]} for r in rows]
