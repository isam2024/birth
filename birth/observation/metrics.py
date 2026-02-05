"""Metrics collection for Birth simulation.

Tracks and logs metrics for observing emergent behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from birth.observation.logger import get_logger

if TYPE_CHECKING:
    from birth.core.engine import SimulationEngine

logger = get_logger("birth.metrics")


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""

    agent_id: str
    agent_name: str
    artworks_created: int = 0
    messages_sent: int = 0
    critiques_written: int = 0
    reflections_generated: int = 0
    avg_mood: float = 0.5
    avg_energy: float = 1.0
    mood_samples: list[float] = field(default_factory=list)
    energy_samples: list[float] = field(default_factory=list)

    def sample(self, mood: float, energy: float) -> None:
        """Record a mood/energy sample."""
        self.mood_samples.append(mood)
        self.energy_samples.append(energy)
        self.avg_mood = sum(self.mood_samples) / len(self.mood_samples)
        self.avg_energy = sum(self.energy_samples) / len(self.energy_samples)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "artworks_created": self.artworks_created,
            "messages_sent": self.messages_sent,
            "critiques_written": self.critiques_written,
            "reflections_generated": self.reflections_generated,
            "avg_mood": self.avg_mood,
            "avg_energy": self.avg_energy,
        }


@dataclass
class SimulationMetrics:
    """Global simulation metrics."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    cycles_completed: int = 0
    total_artworks: int = 0
    total_interactions: int = 0
    total_reflections: int = 0

    # Emergence indicators
    art_style_clusters: dict[str, int] = field(default_factory=dict)
    social_graph_edges: int = 0
    collaboration_count: int = 0

    # Per-agent metrics
    agent_metrics: dict[str, AgentMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "cycles_completed": self.cycles_completed,
            "total_artworks": self.total_artworks,
            "total_interactions": self.total_interactions,
            "total_reflections": self.total_reflections,
            "emergence": {
                "art_style_clusters": self.art_style_clusters,
                "social_graph_edges": self.social_graph_edges,
                "collaboration_count": self.collaboration_count,
            },
            "agents": {
                aid: am.to_dict()
                for aid, am in self.agent_metrics.items()
            },
        }


class MetricsCollector:
    """Collects and tracks simulation metrics."""

    def __init__(self):
        self._metrics = SimulationMetrics()
        self._sample_interval = 10  # cycles between samples

    @property
    def metrics(self) -> SimulationMetrics:
        """Current metrics."""
        return self._metrics

    def register_agent(self, agent_id: str, agent_name: str) -> None:
        """Register a new agent for tracking."""
        if agent_id not in self._metrics.agent_metrics:
            self._metrics.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
            )

    def record_artwork(self, agent_id: str, style_tags: list[str]) -> None:
        """Record artwork creation."""
        self._metrics.total_artworks += 1

        if agent_id in self._metrics.agent_metrics:
            self._metrics.agent_metrics[agent_id].artworks_created += 1

        # Track style clusters
        for tag in style_tags:
            self._metrics.art_style_clusters[tag] = (
                self._metrics.art_style_clusters.get(tag, 0) + 1
            )

    def record_interaction(
        self, from_agent_id: str, to_agent_id: str, interaction_type: str
    ) -> None:
        """Record an interaction."""
        self._metrics.total_interactions += 1

        if interaction_type == "message":
            if from_agent_id in self._metrics.agent_metrics:
                self._metrics.agent_metrics[from_agent_id].messages_sent += 1
        elif interaction_type == "critique":
            if from_agent_id in self._metrics.agent_metrics:
                self._metrics.agent_metrics[from_agent_id].critiques_written += 1
        elif interaction_type == "collaboration":
            self._metrics.collaboration_count += 1

        # Update social graph (simplified - just count unique edges)
        self._metrics.social_graph_edges += 1

    def record_reflection(self, agent_id: str) -> None:
        """Record a reflection."""
        self._metrics.total_reflections += 1

        if agent_id in self._metrics.agent_metrics:
            self._metrics.agent_metrics[agent_id].reflections_generated += 1

    def record_cycle(self, engine: "SimulationEngine") -> None:
        """Record a cycle completion and sample agent states."""
        self._metrics.cycles_completed += 1

        # Sample agent states periodically
        if self._metrics.cycles_completed % self._sample_interval == 0:
            for agent in engine.agents:
                if agent.id in self._metrics.agent_metrics:
                    self._metrics.agent_metrics[agent.id].sample(
                        agent.mood, agent.energy
                    )

    def get_emergence_report(self) -> dict:
        """Generate a report on emergent behaviors.

        Returns:
            Report dictionary
        """
        # Identify dominant styles
        style_counts = self._metrics.art_style_clusters
        dominant_styles = sorted(
            style_counts.items(), key=lambda x: -x[1]
        )[:5]

        # Calculate social density
        agent_count = len(self._metrics.agent_metrics)
        max_edges = agent_count * (agent_count - 1) if agent_count > 1 else 1
        social_density = self._metrics.social_graph_edges / max_edges

        # Find most prolific artists
        prolific = sorted(
            self._metrics.agent_metrics.values(),
            key=lambda a: -a.artworks_created,
        )[:5]

        # Find most social agents
        social = sorted(
            self._metrics.agent_metrics.values(),
            key=lambda a: -(a.messages_sent + a.critiques_written),
        )[:5]

        return {
            "cycles": self._metrics.cycles_completed,
            "dominant_styles": dominant_styles,
            "social_density": social_density,
            "collaborations": self._metrics.collaboration_count,
            "prolific_artists": [
                {"name": a.agent_name, "works": a.artworks_created}
                for a in prolific
            ],
            "social_agents": [
                {"name": a.agent_name, "interactions": a.messages_sent + a.critiques_written}
                for a in social
            ],
        }

    def log_summary(self) -> None:
        """Log a summary of current metrics."""
        report = self.get_emergence_report()
        logger.info(
            "metrics_summary",
            cycles=report["cycles"],
            total_artworks=self._metrics.total_artworks,
            total_interactions=self._metrics.total_interactions,
            dominant_styles=report["dominant_styles"][:3],
            social_density=f"{report['social_density']:.2%}",
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = SimulationMetrics()
