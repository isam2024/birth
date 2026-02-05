"""Data access layer for Birth simulation."""

import json
from datetime import datetime

from birth.storage.database import Database
from birth.storage.models import (
    Agent,
    Artwork,
    Interaction,
    Memory,
    MemoryType,
    Reflection,
    Sentiment,
)


class Repository:
    """Unified data access for all Birth entities."""

    def __init__(self, database: Database):
        self._db = database

    # ========== Agents ==========

    async def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent."""
        await self._db.execute(
            """
            INSERT INTO agents (id, name, backstory, philosophy, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                agent.id,
                agent.name,
                agent.backstory,
                agent.philosophy,
                agent.created_at.isoformat(),
                agent.is_active,
            ),
        )
        return agent

    async def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        row = await self._db.fetch_one(
            "SELECT * FROM agents WHERE id = ?", (agent_id,)
        )
        if row:
            return Agent(
                id=row["id"],
                name=row["name"],
                backstory=row["backstory"],
                philosophy=row["philosophy"],
                created_at=datetime.fromisoformat(row["created_at"]),
                is_active=bool(row["is_active"]),
            )
        return None

    async def get_active_agents(self) -> list[Agent]:
        """Get all active agents."""
        rows = await self._db.fetch_all(
            "SELECT * FROM agents WHERE is_active = 1"
        )
        return [
            Agent(
                id=row["id"],
                name=row["name"],
                backstory=row["backstory"],
                philosophy=row["philosophy"],
                created_at=datetime.fromisoformat(row["created_at"]),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    async def deactivate_agent(self, agent_id: str) -> None:
        """Deactivate an agent."""
        await self._db.execute(
            "UPDATE agents SET is_active = 0 WHERE id = ?", (agent_id,)
        )

    # ========== Memories ==========

    async def add_memory(self, memory: Memory) -> Memory:
        """Add a memory to an agent's memory stream."""
        cursor = await self._db.execute(
            """
            INSERT INTO memories (agent_id, timestamp, memory_type, content, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory.agent_id,
                memory.timestamp.isoformat(),
                memory.memory_type,
                memory.content,
                memory.importance,
                memory.embedding,
            ),
        )
        memory.id = cursor.lastrowid
        return memory

    async def get_recent_memories(
        self, agent_id: str, limit: int = 50
    ) -> list[Memory]:
        """Get recent memories for an agent."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM memories
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_id, limit),
        )
        return [
            Memory(
                id=row["id"],
                agent_id=row["agent_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                memory_type=MemoryType(row["memory_type"]),
                content=row["content"],
                importance=row["importance"],
                embedding=row["embedding"],
            )
            for row in rows
        ]

    async def get_important_memories(
        self, agent_id: str, threshold: float = 0.6, limit: int = 20
    ) -> list[Memory]:
        """Get important memories above a threshold."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM memories
            WHERE agent_id = ? AND importance >= ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
            """,
            (agent_id, threshold, limit),
        )
        return [
            Memory(
                id=row["id"],
                agent_id=row["agent_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                memory_type=MemoryType(row["memory_type"]),
                content=row["content"],
                importance=row["importance"],
                embedding=row["embedding"],
            )
            for row in rows
        ]

    # ========== Sentiments ==========

    async def get_sentiment(
        self, agent_id: str, target_type: str, target_id: str
    ) -> Sentiment | None:
        """Get an agent's sentiment toward a target."""
        row = await self._db.fetch_one(
            """
            SELECT * FROM sentiments
            WHERE agent_id = ? AND target_type = ? AND target_id = ?
            """,
            (agent_id, target_type, target_id),
        )
        if row:
            return Sentiment(
                id=row["id"],
                agent_id=row["agent_id"],
                target_type=row["target_type"],
                target_id=row["target_id"],
                sentiment_score=row["sentiment_score"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )
        return None

    async def upsert_sentiment(self, sentiment: Sentiment) -> Sentiment:
        """Create or update a sentiment."""
        await self._db.execute(
            """
            INSERT INTO sentiments (agent_id, target_type, target_id, sentiment_score, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, target_type, target_id)
            DO UPDATE SET sentiment_score = ?, last_updated = ?
            """,
            (
                sentiment.agent_id,
                sentiment.target_type,
                sentiment.target_id,
                sentiment.sentiment_score,
                sentiment.last_updated.isoformat(),
                sentiment.sentiment_score,
                sentiment.last_updated.isoformat(),
            ),
        )
        return sentiment

    async def get_all_sentiments(self, agent_id: str) -> list[Sentiment]:
        """Get all sentiments for an agent."""
        rows = await self._db.fetch_all(
            "SELECT * FROM sentiments WHERE agent_id = ?", (agent_id,)
        )
        return [
            Sentiment(
                id=row["id"],
                agent_id=row["agent_id"],
                target_type=row["target_type"],
                target_id=row["target_id"],
                sentiment_score=row["sentiment_score"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )
            for row in rows
        ]

    async def decay_all_sentiments(self, agent_id: str, rate: float = 0.01) -> None:
        """Decay all sentiments for an agent toward neutral."""
        await self._db.execute(
            """
            UPDATE sentiments
            SET sentiment_score = CASE
                WHEN sentiment_score > 0 THEN MAX(0, sentiment_score - ?)
                WHEN sentiment_score < 0 THEN MIN(0, sentiment_score + ?)
                ELSE 0
            END,
            last_updated = ?
            WHERE agent_id = ?
            """,
            (rate, rate, datetime.utcnow().isoformat(), agent_id),
        )

    # ========== Artworks ==========

    async def create_artwork(self, artwork: Artwork) -> Artwork:
        """Create a new artwork."""
        await self._db.execute(
            """
            INSERT INTO artworks (id, creator_id, title, medium, content_text,
                                  image_path, style_tags, created_at, inspiration_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artwork.id,
                artwork.creator_id,
                artwork.title,
                artwork.medium,
                artwork.content_text,
                artwork.image_path,
                json.dumps(artwork.style_tags),
                artwork.created_at.isoformat(),
                json.dumps(artwork.inspiration_ids),
            ),
        )
        return artwork

    async def get_artwork(self, artwork_id: str) -> Artwork | None:
        """Get an artwork by ID."""
        row = await self._db.fetch_one(
            "SELECT * FROM artworks WHERE id = ?", (artwork_id,)
        )
        if row:
            return Artwork(
                id=row["id"],
                creator_id=row["creator_id"],
                title=row["title"],
                medium=row["medium"],
                content_text=row["content_text"],
                image_path=row["image_path"],
                style_tags=json.loads(row["style_tags"] or "[]"),
                created_at=datetime.fromisoformat(row["created_at"]),
                inspiration_ids=json.loads(row["inspiration_ids"] or "[]"),
            )
        return None

    async def get_recent_artworks(self, limit: int = 20) -> list[Artwork]:
        """Get recent artworks from the Commons."""
        rows = await self._db.fetch_all(
            "SELECT * FROM artworks ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [
            Artwork(
                id=row["id"],
                creator_id=row["creator_id"],
                title=row["title"],
                medium=row["medium"],
                content_text=row["content_text"],
                image_path=row["image_path"],
                style_tags=json.loads(row["style_tags"] or "[]"),
                created_at=datetime.fromisoformat(row["created_at"]),
                inspiration_ids=json.loads(row["inspiration_ids"] or "[]"),
            )
            for row in rows
        ]

    async def get_artworks_by_creator(self, creator_id: str) -> list[Artwork]:
        """Get all artworks by a specific creator."""
        rows = await self._db.fetch_all(
            "SELECT * FROM artworks WHERE creator_id = ? ORDER BY created_at DESC",
            (creator_id,),
        )
        return [
            Artwork(
                id=row["id"],
                creator_id=row["creator_id"],
                title=row["title"],
                medium=row["medium"],
                content_text=row["content_text"],
                image_path=row["image_path"],
                style_tags=json.loads(row["style_tags"] or "[]"),
                created_at=datetime.fromisoformat(row["created_at"]),
                inspiration_ids=json.loads(row["inspiration_ids"] or "[]"),
            )
            for row in rows
        ]

    # ========== Reflections ==========

    async def add_reflection(self, reflection: Reflection) -> Reflection:
        """Add a reflection for an agent."""
        cursor = await self._db.execute(
            """
            INSERT INTO reflections (agent_id, timestamp, insight, derived_from)
            VALUES (?, ?, ?, ?)
            """,
            (
                reflection.agent_id,
                reflection.timestamp.isoformat(),
                reflection.insight,
                json.dumps(reflection.derived_from),
            ),
        )
        reflection.id = cursor.lastrowid
        return reflection

    async def get_recent_reflections(
        self, agent_id: str, limit: int = 10
    ) -> list[Reflection]:
        """Get recent reflections for an agent."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM reflections
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_id, limit),
        )
        return [
            Reflection(
                id=row["id"],
                agent_id=row["agent_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                insight=row["insight"],
                derived_from=json.loads(row["derived_from"] or "[]"),
            )
            for row in rows
        ]

    # ========== Interactions ==========

    async def add_interaction(self, interaction: Interaction) -> Interaction:
        """Record a social interaction."""
        cursor = await self._db.execute(
            """
            INSERT INTO interactions (from_agent_id, to_agent_id, interaction_type, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                interaction.from_agent_id,
                interaction.to_agent_id,
                interaction.interaction_type,
                interaction.content,
                interaction.timestamp.isoformat(),
            ),
        )
        interaction.id = cursor.lastrowid
        return interaction

    async def get_interactions_for_agent(
        self, agent_id: str, limit: int = 50
    ) -> list[Interaction]:
        """Get interactions involving an agent."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM interactions
            WHERE from_agent_id = ? OR to_agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_id, agent_id, limit),
        )
        return [
            Interaction(
                id=row["id"],
                from_agent_id=row["from_agent_id"],
                to_agent_id=row["to_agent_id"],
                interaction_type=row["interaction_type"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]
