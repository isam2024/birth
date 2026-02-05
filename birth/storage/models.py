"""Data models for Birth simulation."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories an agent can have."""

    PERCEPTION = "perception"  # Something observed
    ACTION = "action"  # Something done
    THOUGHT = "thought"  # Internal processing
    REFLECTION = "reflection"  # High-level insight


class Agent(BaseModel):
    """An autonomous artist agent."""

    id: str
    name: str
    backstory: str
    philosophy: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    # Runtime state (not persisted directly)
    mood: float = 0.5  # 0.0 = very negative, 1.0 = very positive
    energy: float = 1.0  # 0.0 = exhausted, 1.0 = fully rested


class Memory(BaseModel):
    """A single memory in an agent's memory stream."""

    id: int | None = None
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: MemoryType
    content: str
    importance: float = 0.5  # 0.0 = trivial, 1.0 = crucial
    embedding: bytes | None = None  # For future semantic search

    class Config:
        use_enum_values = True


class Sentiment(BaseModel):
    """An agent's feeling toward a target."""

    id: int | None = None
    agent_id: str
    target_type: str  # 'agent', 'artwork', 'concept'
    target_id: str
    sentiment_score: float = 0.0  # -1.0 = hatred, +1.0 = love
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def adjust(self, delta: float, weight: float = 0.3) -> None:
        """Adjust sentiment score with weighted update."""
        self.sentiment_score = max(-1.0, min(1.0, self.sentiment_score + delta * weight))
        self.last_updated = datetime.utcnow()

    def decay(self, rate: float = 0.01) -> None:
        """Decay sentiment toward neutral over time."""
        if self.sentiment_score > 0:
            self.sentiment_score = max(0, self.sentiment_score - rate)
        elif self.sentiment_score < 0:
            self.sentiment_score = min(0, self.sentiment_score + rate)


class Artwork(BaseModel):
    """A piece of art created by an agent."""

    id: str
    creator_id: str
    title: str
    medium: str  # 'text', 'image', 'mixed'
    content_text: str | None = None
    image_path: str | None = None
    style_tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    inspiration_ids: list[str] = Field(default_factory=list)  # IDs of inspiring artworks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "creator_id": self.creator_id,
            "title": self.title,
            "medium": self.medium,
            "content_text": self.content_text,
            "image_path": self.image_path,
            "style_tags": self.style_tags,
            "created_at": self.created_at,
            "inspiration_ids": self.inspiration_ids,
        }


class Reflection(BaseModel):
    """A high-level insight derived from memories."""

    id: int | None = None
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    insight: str
    derived_from: list[int] = Field(default_factory=list)  # Memory IDs


class Interaction(BaseModel):
    """A social interaction between agents."""

    id: int | None = None
    from_agent_id: str
    to_agent_id: str
    interaction_type: str  # 'message', 'critique', 'praise', 'collaboration_invite'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
