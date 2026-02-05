"""Sentiment and relationship model for agents.

The Heart - this is the core of intrinsic motivation and the antidote to blandness.
It transforms neutral observations into biased experiences.
"""

from datetime import datetime

from birth.config import get_config
from birth.observation.logger import AgentLogger
from birth.storage.models import Sentiment
from birth.storage.repository import Repository


class SentimentModel:
    """Manages an agent's feelings toward entities in the world.

    Tracks feelings toward:
    - Other agents
    - Artworks
    - Abstract concepts (themes, styles)
    """

    def __init__(
        self,
        agent_id: str,
        repository: Repository,
        logger: AgentLogger,
    ):
        self._agent_id = agent_id
        self._repository = repository
        self._logger = logger
        self._config = get_config().simulation

        # In-memory cache of sentiments
        self._cache: dict[tuple[str, str], Sentiment] = {}

        # Name cache for readable summaries (maps IDs to display names)
        self._name_cache: dict[str, str] = {}

    def register_name(self, entity_id: str, name: str) -> None:
        """Register a display name for an entity ID."""
        self._name_cache[entity_id] = name

    async def load(self) -> None:
        """Load all sentiments from database."""
        sentiments = await self._repository.get_all_sentiments(self._agent_id)
        self._cache = {
            (s.target_type, s.target_id): s for s in sentiments
        }

    async def get(self, target_type: str, target_id: str) -> float:
        """Get sentiment toward a target.

        Args:
            target_type: 'agent', 'artwork', or 'concept'
            target_id: Target identifier

        Returns:
            Sentiment score (-1.0 to +1.0), 0.0 if unknown
        """
        key = (target_type, target_id)
        if key in self._cache:
            return self._cache[key].sentiment_score
        return 0.0

    async def update(
        self,
        target_type: str,
        target_id: str,
        delta: float,
        weight: float | None = None,
    ) -> float:
        """Update sentiment toward a target.

        Args:
            target_type: 'agent', 'artwork', or 'concept'
            target_id: Target identifier
            delta: Change in sentiment (-1.0 to +1.0)
            weight: Update weight (default from config)

        Returns:
            New sentiment score
        """
        weight = weight or self._config.sentiment_update_weight
        key = (target_type, target_id)

        # Get or create sentiment
        if key in self._cache:
            sentiment = self._cache[key]
            old_score = sentiment.sentiment_score
        else:
            sentiment = Sentiment(
                agent_id=self._agent_id,
                target_type=target_type,
                target_id=target_id,
                sentiment_score=0.0,
            )
            old_score = 0.0

        # Apply weighted update
        sentiment.adjust(delta, weight)

        # Save to database
        await self._repository.upsert_sentiment(sentiment)
        self._cache[key] = sentiment

        # Log significant changes
        if abs(sentiment.sentiment_score - old_score) > 0.05:
            self._logger.sentiment_change(
                target_type, target_id, old_score, sentiment.sentiment_score
            )

        return sentiment.sentiment_score

    async def set(self, target_type: str, target_id: str, score: float) -> None:
        """Set sentiment to a specific value.

        Args:
            target_type: 'agent', 'artwork', or 'concept'
            target_id: Target identifier
            score: New sentiment score (-1.0 to +1.0)
        """
        key = (target_type, target_id)
        score = max(-1.0, min(1.0, score))

        if key in self._cache:
            sentiment = self._cache[key]
            old_score = sentiment.sentiment_score
            sentiment.sentiment_score = score
            sentiment.last_updated = datetime.utcnow()
        else:
            sentiment = Sentiment(
                agent_id=self._agent_id,
                target_type=target_type,
                target_id=target_id,
                sentiment_score=score,
            )
            old_score = 0.0

        await self._repository.upsert_sentiment(sentiment)
        self._cache[key] = sentiment

        self._logger.sentiment_change(target_type, target_id, old_score, score)

    async def decay_all(self) -> None:
        """Decay all sentiments toward neutral.

        Called periodically to simulate forgetting/mellowing.
        """
        rate = self._config.sentiment_decay_rate
        await self._repository.decay_all_sentiments(self._agent_id, rate)

        # Update cache
        for sentiment in self._cache.values():
            sentiment.decay(rate)

    def get_toward_agent(self, agent_id: str) -> float:
        """Get sentiment toward another agent."""
        return self._cache.get(("agent", agent_id), Sentiment(
            agent_id=self._agent_id,
            target_type="agent",
            target_id=agent_id,
        )).sentiment_score

    def get_toward_artwork(self, artwork_id: str) -> float:
        """Get sentiment toward an artwork."""
        return self._cache.get(("artwork", artwork_id), Sentiment(
            agent_id=self._agent_id,
            target_type="artwork",
            target_id=artwork_id,
        )).sentiment_score

    def get_toward_concept(self, concept: str) -> float:
        """Get sentiment toward a concept."""
        return self._cache.get(("concept", concept), Sentiment(
            agent_id=self._agent_id,
            target_type="concept",
            target_id=concept,
        )).sentiment_score

    def get_friends(self, threshold: float = 0.3) -> list[str]:
        """Get agent IDs that this agent feels positively toward.

        Args:
            threshold: Minimum positive sentiment

        Returns:
            List of agent IDs
        """
        return [
            s.target_id
            for s in self._cache.values()
            if s.target_type == "agent" and s.sentiment_score >= threshold
        ]

    def get_rivals(self, threshold: float = -0.3) -> list[str]:
        """Get agent IDs that this agent feels negatively toward.

        Args:
            threshold: Maximum negative sentiment

        Returns:
            List of agent IDs
        """
        return [
            s.target_id
            for s in self._cache.values()
            if s.target_type == "agent" and s.sentiment_score <= threshold
        ]

    def summarize(self, name_cache: dict[str, str] | None = None) -> str:
        """Create a text summary of significant sentiments.

        Args:
            name_cache: Optional dict mapping IDs to display names

        Returns:
            Formatted summary for prompts
        """
        name_cache = name_cache or self._name_cache
        lines = []

        def get_name(target_type: str, target_id: str) -> str:
            """Resolve ID to readable name."""
            if target_id in name_cache:
                return name_cache[target_id]
            # For concepts like "drop:abc123", extract meaningful part
            if target_type == "concept" and ":" in target_id:
                return target_id.split(":", 1)[1][:20]
            return target_id[:12]  # Truncate UUIDs

        # Strong positive sentiments
        positive = [s for s in self._cache.values() if s.sentiment_score >= 0.3]
        if positive:
            for s in sorted(positive, key=lambda x: -x.sentiment_score)[:5]:
                name = get_name(s.target_type, s.target_id)
                if s.target_type == "agent":
                    lines.append(f"I admire {name}'s work")
                elif s.target_type == "artwork":
                    lines.append(f"I was moved by '{name}'")
                elif s.target_type == "concept":
                    lines.append(f"I'm drawn to {name}")

        # Strong negative sentiments
        negative = [s for s in self._cache.values() if s.sentiment_score <= -0.3]
        if negative:
            for s in sorted(negative, key=lambda x: x.sentiment_score)[:5]:
                name = get_name(s.target_type, s.target_id)
                if s.target_type == "agent":
                    lines.append(f"I find {name}'s work uninspiring")
                elif s.target_type == "artwork":
                    lines.append(f"I was unmoved by '{name}'")
                elif s.target_type == "concept":
                    lines.append(f"I'm repelled by {name}")

        return "\n".join(lines) if lines else ""

    @property
    def overall_mood_modifier(self) -> float:
        """Calculate mood modifier from sentiments.

        Positive sentiments boost mood, negative ones lower it.

        Returns:
            Mood modifier (-0.2 to +0.2)
        """
        if not self._cache:
            return 0.0

        total = sum(s.sentiment_score for s in self._cache.values())
        avg = total / len(self._cache)

        # Scale to reasonable mood impact
        return max(-0.2, min(0.2, avg * 0.3))
