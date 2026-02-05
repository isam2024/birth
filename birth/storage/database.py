"""SQLite database management for Birth simulation."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from birth.config import Config, get_config

SCHEMA = """
-- Agents
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    backstory TEXT,
    philosophy TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Memory Stream
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT REFERENCES agents(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_type TEXT,
    content TEXT,
    importance REAL DEFAULT 0.5,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

-- Sentiment Model
CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT REFERENCES agents(id),
    target_type TEXT,
    target_id TEXT,
    sentiment_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, target_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_sentiments_agent_id ON sentiments(agent_id);

-- Artworks
CREATE TABLE IF NOT EXISTS artworks (
    id TEXT PRIMARY KEY,
    creator_id TEXT REFERENCES agents(id),
    title TEXT,
    medium TEXT,
    content_text TEXT,
    image_path TEXT,
    style_tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inspiration_ids TEXT
);

CREATE INDEX IF NOT EXISTS idx_artworks_creator_id ON artworks(creator_id);
CREATE INDEX IF NOT EXISTS idx_artworks_created_at ON artworks(created_at);

-- Agent Reflections
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT REFERENCES agents(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    insight TEXT,
    derived_from TEXT
);

CREATE INDEX IF NOT EXISTS idx_reflections_agent_id ON reflections(agent_id);

-- Social Interactions
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_agent_id TEXT REFERENCES agents(id),
    to_agent_id TEXT REFERENCES agents(id),
    interaction_type TEXT,
    content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_interactions_from ON interactions(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_interactions_to ON interactions(to_agent_id);
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path | None = None):
        self._config = get_config()
        self._db_path = db_path or self._config.database_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        """Get database file path."""
        return self._db_path

    async def connect(self) -> None:
        """Establish database connection and initialize schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.executescript(SCHEMA)
        await self._connection.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for database transactions."""
        async with self._lock:
            if not self._connection:
                raise RuntimeError("Database not connected")
            try:
                yield self._connection
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise

    async def execute(
        self, query: str, params: tuple | dict | None = None
    ) -> aiosqlite.Cursor:
        """Execute a single query."""
        async with self._lock:
            if not self._connection:
                raise RuntimeError("Database not connected")
            cursor = await self._connection.execute(query, params or ())
            await self._connection.commit()
            return cursor

    async def execute_many(self, query: str, params_list: list[tuple | dict]) -> None:
        """Execute query with multiple parameter sets."""
        async with self._lock:
            if not self._connection:
                raise RuntimeError("Database not connected")
            await self._connection.executemany(query, params_list)
            await self._connection.commit()

    async def fetch_one(
        self, query: str, params: tuple | dict | None = None
    ) -> aiosqlite.Row | None:
        """Fetch a single row."""
        async with self._lock:
            if not self._connection:
                raise RuntimeError("Database not connected")
            cursor = await self._connection.execute(query, params or ())
            return await cursor.fetchone()

    async def fetch_all(
        self, query: str, params: tuple | dict | None = None
    ) -> list[aiosqlite.Row]:
        """Fetch all rows."""
        async with self._lock:
            if not self._connection:
                raise RuntimeError("Database not connected")
            cursor = await self._connection.execute(query, params or ())
            return await cursor.fetchall()


# Global database instance
_database: Database | None = None


async def get_database(config: Config | None = None) -> Database:
    """Get or create the global database instance."""
    global _database
    if _database is None:
        cfg = config or get_config()
        _database = Database(cfg.database_path)
        await _database.connect()
    return _database


async def close_database() -> None:
    """Close the global database instance."""
    global _database
    if _database:
        await _database.close()
        _database = None
