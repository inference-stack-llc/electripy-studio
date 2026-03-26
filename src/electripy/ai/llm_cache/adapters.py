"""Cache backend adapters for the LLM Caching Layer."""

from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict

from .domain import CacheEntry, CacheStats


class InMemoryCacheBackend:
    """Thread-safe in-memory LRU cache backend.

    Args:
      max_size: Maximum number of entries before LRU eviction.
    """

    __slots__ = ("_max_size", "_lock", "_store", "_hits", "_misses")

    def __init__(self, *, max_size: int = 1024) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cached entry, promoting it to most-recently-used."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._store.move_to_end(key)
            # Return with incremented hit count
            updated = CacheEntry(
                key=entry.key,
                response_text=entry.response_text,
                model=entry.model,
                hit_count=entry.hit_count + 1,
            )
            self._store[key] = updated
            return updated

    def put(self, entry: CacheEntry) -> None:
        """Store an entry, evicting LRU if at capacity."""
        with self._lock:
            if entry.key in self._store:
                self._store.move_to_end(entry.key)
                self._store[entry.key] = entry
                return
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[entry.key] = entry

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._store),
            )

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


class SqliteCacheBackend:
    """SQLite-backed persistent cache.

    Process-safe via SQLite's built-in locking.  Uses WAL mode for
    concurrent reads.

    Args:
      db_path: Path to the SQLite database file.  Use ``:memory:`` for
        an in-process ephemeral store.
      max_size: Maximum number of entries before oldest-first eviction.
    """

    __slots__ = ("_conn", "_max_size", "_hits", "_misses", "_lock")

    def __init__(self, *, db_path: str = ":memory:", max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  response_text TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  hit_count INTEGER NOT NULL DEFAULT 0,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve and bump hit counter."""
        with self._lock:
            row = self._conn.execute(
                "SELECT key, response_text, model, hit_count FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
            new_count = row[3] + 1
            self._conn.execute("UPDATE cache SET hit_count = ? WHERE key = ?", (new_count, key))
            self._conn.commit()
            return CacheEntry(key=row[0], response_text=row[1], model=row[2], hit_count=new_count)

    def put(self, entry: CacheEntry) -> None:
        """Insert or replace an entry, evicting oldest if at capacity."""
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count >= self._max_size:
                self._conn.execute(
                    "DELETE FROM cache WHERE key = ("
                    "  SELECT key FROM cache ORDER BY created_at ASC LIMIT 1"
                    ")"
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, response_text, model, hit_count) "
                "VALUES (?, ?, ?, ?)",
                (entry.key, entry.response_text, entry.model, entry.hit_count),
            )
            self._conn.commit()

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            size = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            return CacheStats(hits=self._hits, misses=self._misses, size=size)

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            self._hits = 0
            self._misses = 0
