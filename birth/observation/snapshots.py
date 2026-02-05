"""Agent snapshot system for tracking evolution over time."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from birth.config import get_config
from birth.observation.logger import get_logger

if TYPE_CHECKING:
    from birth.agents.agent import AutonomousAgent

logger = get_logger("birth.snapshots")


class AgentSnapshot:
    """Captures an agent's state at a point in time."""

    def __init__(self, agent: "AutonomousAgent", cycle: int):
        self.agent_id = agent.id
        self.agent_name = agent.name
        self.cycle = cycle
        self.timestamp = datetime.utcnow()

        # Core identity
        self.philosophy = agent.philosophy
        self.backstory = agent.backstory

        # Current state
        self.mood = agent.mood
        self.energy = agent.energy

        # Relationships
        self.friends = agent._sentiment.get_friends(threshold=0.2)
        self.rivals = agent._sentiment.get_rivals(threshold=-0.2)
        self.sentiment_summary = agent._sentiment.summarize()

        # Recent memories (last 10)
        recent_memories = agent._memory.get_recent(10)
        self.recent_memories = [
            {"type": m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value,
             "content": m.content[:200],
             "importance": m.importance}
            for m in recent_memories
        ]

        # Artworks created (populated by SnapshotManager)
        self.artworks_created = []
        self.style_tags_used = []
        self.subjects = []  # Subject matter from artworks

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "cycle": self.cycle,
            "timestamp": self.timestamp.isoformat(),
            "philosophy": self.philosophy,
            "backstory": self.backstory,
            "mood": self.mood,
            "energy": self.energy,
            "friends_count": len(self.friends),
            "rivals_count": len(self.rivals),
            "sentiment_summary": self.sentiment_summary,
            "recent_memories": self.recent_memories,
            "artworks": self.artworks_created,
            "subjects": self.subjects,
            "style_tags_used": self.style_tags_used,
        }

    def to_markdown(self) -> str:
        """Generate a readable markdown snapshot."""
        lines = [
            f"# {self.agent_name}",
            f"*Snapshot at cycle {self.cycle} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Identity",
            f"**Philosophy:** {self.philosophy}",
            "",
            f"**Backstory:** {self.backstory}",
            "",
            "## Current State",
            f"- Mood: {self.mood:.2f}",
            f"- Energy: {self.energy:.2f}",
            f"- Friends: {len(self.friends)}",
            f"- Rivals: {len(self.rivals)}",
            "",
        ]

        if self.sentiment_summary:
            lines.extend([
                "## Feelings",
                self.sentiment_summary,
                "",
            ])

        lines.extend(["## Artworks Created", ""])
        if self.artworks_created:
            for art in self.artworks_created:
                tags = ", ".join(art.get('style_tags', [])[:3]) if art.get('style_tags') else ""
                lines.append(f"- **{art['title']}** ({art['medium']}) {f'[{tags}]' if tags else ''}")
            lines.append("")

            if self.subjects:
                lines.extend([
                    "## Subject Matter",
                    ", ".join(self.subjects),
                    "",
                ])

            if self.style_tags_used:
                lines.extend([
                    "## Styles Used",
                    ", ".join(self.style_tags_used),
                    "",
                ])
        else:
            lines.extend(["*No artworks created yet - this agent has been observing.*", ""])

        if self.recent_memories:
            lines.extend([
                "## Recent Experiences",
            ])
            for m in self.recent_memories[:5]:
                lines.append(f"- [{m['type']}] {m['content'][:100]}...")
            lines.append("")

        return "\n".join(lines)


class SnapshotManager:
    """Manages periodic snapshots of all agents."""

    def __init__(self, snapshot_dir: Path | None = None):
        config = get_config()
        self._snapshot_dir = snapshot_dir or config.output_dir / "snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Track evolution data per agent
        self._evolution: dict[str, list[dict]] = {}

    async def take_snapshot(
        self,
        agents: list["AutonomousAgent"],
        cycle: int,
        repository=None,
    ) -> None:
        """Take snapshots of all agents."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        for agent in agents:
            snapshot = AgentSnapshot(agent, cycle)

            # Get artworks created by this agent
            if repository:
                artworks = await repository.get_artworks_by_creator(agent.id)
                snapshot.artworks_created = [
                    {
                        "title": a.title,
                        "medium": a.medium,
                        "style_tags": a.style_tags,
                        "created_at": a.created_at.isoformat() if a.created_at else None,
                    }
                    for a in artworks[-10:]  # Last 10
                ]

                # Collect all style tags used
                all_tags = []
                for a in artworks:
                    all_tags.extend(a.style_tags or [])
                snapshot.style_tags_used = list(set(all_tags))

                # Extract subjects from titles and content
                subjects = set()
                for a in artworks:
                    # Add title words (excluding common words)
                    if a.title and a.title != "Untitled":
                        subjects.add(a.title)
                    # Add meaningful style tags as subjects
                    for tag in (a.style_tags or []):
                        if tag not in ["visual", "text", "mixed"]:
                            subjects.add(tag)
                snapshot.subjects = list(subjects)[:20]  # Limit to 20

            # Track evolution
            if agent.id not in self._evolution:
                self._evolution[agent.id] = []

            self._evolution[agent.id].append({
                "cycle": cycle,
                "timestamp": snapshot.timestamp.isoformat(),
                "agent_name": agent.name,
                "mood": snapshot.mood,
                "energy": snapshot.energy,
                "friends_count": len(snapshot.friends),
                "rivals_count": len(snapshot.rivals),
                "artwork_count": len(snapshot.artworks_created),
                "subjects": snapshot.subjects,
                "style_tags": snapshot.style_tags_used,
            })

            # Save individual snapshot
            safe_name = self._sanitize_name(agent.name)
            agent_dir = self._snapshot_dir / safe_name
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Save markdown snapshot
            md_path = agent_dir / f"snapshot_{timestamp}.md"
            md_path.write_text(snapshot.to_markdown())

            # Save JSON snapshot
            json_path = agent_dir / f"snapshot_{timestamp}.json"
            json_path.write_text(json.dumps(snapshot.to_dict(), indent=2))

        logger.info("snapshots_taken", agent_count=len(agents), cycle=cycle)

    def save_evolution_summary(self) -> None:
        """Save evolution summaries for all tracked agents."""
        for agent_id, history in self._evolution.items():
            if not history:
                continue

            # Find agent name from first entry or use ID
            agent_name = agent_id[:12]
            for h in history:
                if "agent_name" in h:
                    agent_name = h["agent_name"]
                    break

            safe_name = self._sanitize_name(agent_name)
            agent_dir = self._snapshot_dir / safe_name
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Save evolution JSON
            evolution_path = agent_dir / "evolution.json"
            evolution_path.write_text(json.dumps(history, indent=2))

            # Generate evolution markdown
            self._write_evolution_markdown(agent_dir, agent_name, history)

    def _write_evolution_markdown(
        self,
        agent_dir: Path,
        agent_name: str,
        history: list[dict],
    ) -> None:
        """Write a markdown summary of agent evolution."""
        lines = [
            f"# Evolution of {agent_name}",
            "",
            "## Timeline",
            "",
            "| Cycle | Mood | Energy | Friends | Rivals | Artworks |",
            "|-------|------|--------|---------|--------|----------|",
        ]

        for h in history:
            lines.append(
                f"| {h['cycle']} | {h['mood']:.2f} | {h['energy']:.2f} | "
                f"{h['friends_count']} | {h['rivals_count']} | {h['artwork_count']} |"
            )

        lines.extend(["", "## Style Evolution", ""])

        # Track subject matter evolution
        all_subjects = []
        for h in history:
            all_subjects.extend(h.get("subjects", []))

        if all_subjects:
            # Count frequency
            from collections import Counter
            subject_counts = Counter(all_subjects)
            top_subjects = subject_counts.most_common(15)
            lines.append("**Subjects explored:** " + ", ".join(
                f"{subj} ({count})" for subj, count in top_subjects
            ))
        else:
            lines.append("No subjects recorded yet.")

        lines.extend(["", "## Style Evolution", ""])

        # Track style tag changes
        all_tags = set()
        for h in history:
            all_tags.update(h.get("style_tags", []))

        if all_tags:
            lines.append("**Styles used:** " + ", ".join(sorted(all_tags)))
        else:
            lines.append("No style tags recorded yet.")

        evolution_md = agent_dir / "evolution.md"
        evolution_md.write_text("\n".join(lines))

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for filesystem."""
        import re
        sanitized = re.sub(r'[^\w\-]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')
