"""Memory stream implementation for agents."""

from datetime import datetime
from typing import TYPE_CHECKING

from birth.config import get_config
from birth.observation.logger import get_logger
from birth.storage.models import Memory, MemoryType
from birth.storage.repository import Repository

if TYPE_CHECKING:
    from birth.agents.agent import AutonomousAgent

logger = get_logger("birth.memory")


class MemoryStream:
    """Manages an agent's chronological memory stream.

    The memory stream is the raw, unedited footage of an agent's existence -
    everything they perceive, do, and think.
    """

    def __init__(self, agent_id: str, repository: Repository):
        self._agent_id = agent_id
        self._repository = repository
        self._config = get_config().simulation

        # In-memory cache of recent memories for quick access
        self._cache: list[Memory] = []
        self._cache_limit = self._config.recent_memory_limit

    async def load_recent(self) -> None:
        """Load recent memories from database into cache."""
        self._cache = await self._repository.get_recent_memories(
            self._agent_id, self._cache_limit
        )
        logger.debug(
            "memory_cache_loaded",
            agent_id=self._agent_id,
            count=len(self._cache),
        )

    async def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
    ) -> Memory:
        """Add a new memory to the stream.

        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance score (0.0-1.0)

        Returns:
            The created Memory
        """
        memory = Memory(
            agent_id=self._agent_id,
            memory_type=memory_type,
            content=content,
            importance=min(1.0, max(0.0, importance)),
            timestamp=datetime.utcnow(),
        )

        memory = await self._repository.add_memory(memory)

        # Update cache
        self._cache.insert(0, memory)
        if len(self._cache) > self._cache_limit:
            self._cache = self._cache[: self._cache_limit]

        # Get type value (handle both enum and string)
        type_val = memory_type.value if hasattr(memory_type, 'value') else memory_type
        logger.debug(
            "memory_added",
            agent_id=self._agent_id,
            type=type_val,
            importance=importance,
        )

        return memory

    async def perceive(self, content: str, importance: float = 0.5) -> Memory:
        """Record a perception."""
        return await self.add(content, MemoryType.PERCEPTION, importance)

    async def act(self, content: str, importance: float = 0.6) -> Memory:
        """Record an action taken."""
        return await self.add(content, MemoryType.ACTION, importance)

    async def think(self, content: str, importance: float = 0.4) -> Memory:
        """Record an internal thought."""
        return await self.add(content, MemoryType.THOUGHT, importance)

    async def reflect(self, content: str, importance: float = 0.8) -> Memory:
        """Record a reflection/insight."""
        return await self.add(content, MemoryType.REFLECTION, importance)

    def get_recent(self, limit: int | None = None) -> list[Memory]:
        """Get recent memories from cache.

        Args:
            limit: Max memories to return (default: all cached)

        Returns:
            List of memories, most recent first
        """
        if limit is None:
            return self._cache.copy()
        return self._cache[:limit]

    async def get_important(
        self,
        threshold: float | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Get important memories above threshold.

        Args:
            threshold: Minimum importance (default from config)
            limit: Max memories to return

        Returns:
            List of important memories
        """
        threshold = threshold or self._config.memory_importance_threshold
        return await self._repository.get_important_memories(
            self._agent_id, threshold, limit
        )

    def get_by_type(self, memory_type: MemoryType, limit: int = 20) -> list[Memory]:
        """Get recent memories of a specific type.

        Args:
            memory_type: Type to filter by
            limit: Max memories to return

        Returns:
            Filtered list of memories
        """
        return [m for m in self._cache if m.memory_type == memory_type][:limit]

    def summarize_recent(self, limit: int = 10) -> str:
        """Create a text summary of recent memories.

        Args:
            limit: Number of memories to include

        Returns:
            Formatted summary string
        """
        memories = self.get_recent(limit)
        if not memories:
            return "No recent memories."

        lines = []
        for m in memories:
            time_str = m.timestamp.strftime("%H:%M")
            # Handle both enum and string (pydantic use_enum_values)
            type_val = m.memory_type.value if hasattr(m.memory_type, 'value') else m.memory_type
            type_str = type_val.upper()
            lines.append(f"[{time_str}] ({type_str}) {m.content}")

        return "\n".join(lines)

    def get_memory_ids(self, limit: int = 10) -> list[int]:
        """Get IDs of recent memories (for reflection tracking).

        Args:
            limit: Number of memory IDs to return

        Returns:
            List of memory IDs
        """
        return [m.id for m in self._cache[:limit] if m.id is not None]

    @property
    def count(self) -> int:
        """Number of memories in cache."""
        return len(self._cache)

    @property
    def latest(self) -> Memory | None:
        """Most recent memory, if any."""
        return self._cache[0] if self._cache else None
