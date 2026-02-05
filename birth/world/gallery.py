"""Gallery - Emergent repository of inspirational assets.

The Gallery evolves over time through agent contributions:
- Peer-nominated: Highly praised works get added
- Style evolution: Frequently used tags become concepts

Initial seed content provides artistic vocabulary, but the gallery
grows organically from what agents create and value.
"""

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from birth.config import get_config
from birth.observation.logger import get_logger

logger = get_logger("birth.gallery")


# Artistic vocabulary and thematic grounding for coherent aesthetics
DEFAULT_INSPIRATIONS = [
    # Nature
    "The way morning light fractures through mist over a still lake.",
    "A single leaf falling in an empty forest - does it make art?",
    "The geometry of honeycombs: nature's answer to efficiency.",
    "Storm clouds gathering at dusk, violet and amber warnings.",
    "The patience of mountains, wearing down grain by grain.",

    # Emotion
    "The weight of an unspoken word between old friends.",
    "Joy so sharp it feels like grief in reverse.",
    "The hollow echo of footsteps in an empty house.",
    "That moment of suspension before tears arrive.",
    "Love that outlives the lovers themselves.",

    # Philosophy
    "If you could paint silence, what color would it be?",
    "The space between heartbeats contains multitudes.",
    "Every ending is a door pretending to be a wall.",
    "We are all translations of something that has no original.",
    "The universe experiencing itself through temporary eyes.",

    # Art
    "A canvas that refuses to hold the paint.",
    "The conversation between negative space and form.",
    "When the frame becomes part of the work.",
    "Art that exists only in the moment of its destruction.",
    "The ghost of the artist's hand in every brushstroke.",

    # Memory
    "The smell of a place you've never been but somehow remember.",
    "Photographs that remember more than you do.",
    "The way certain songs are time machines.",
    "Dreams that feel more real than waking.",
    "Nostalgia for futures that never happened.",

    # Paradox
    "The more you look, the less you see.",
    "Perfect imperfection.",
    "The loudness of absolute silence.",
    "Finding home in displacement.",
    "The weight of emptiness.",
]

DEFAULT_CONCEPTS = [
    {"theme": "transience", "elements": ["water", "light", "shadow", "time"]},
    {"theme": "connection", "elements": ["threads", "bridges", "mirrors", "echoes"]},
    {"theme": "solitude", "elements": ["empty rooms", "single objects", "vast spaces"]},
    {"theme": "transformation", "elements": ["metamorphosis", "decay", "growth", "fire"]},
    {"theme": "duality", "elements": ["light/dark", "presence/absence", "sound/silence"]},
    {"theme": "memory", "elements": ["fragments", "fading", "layers", "traces"]},
    {"theme": "chaos", "elements": ["entropy", "storms", "fractals", "unraveling"]},
    {"theme": "order", "elements": ["geometry", "patterns", "symmetry", "grids"]},
]


class Gallery:
    """Repository of inspirational content for agents."""

    def __init__(self, gallery_dir: Path | None = None):
        config = get_config()
        self._dir = gallery_dir or config.gallery_dir
        self._texts: list[str] = []
        self._concepts: list[dict] = []
        self._loaded = False

    async def load(self) -> None:
        """Load inspirational content from files."""
        self._texts = []
        self._concepts = []

        # Load text files
        texts_dir = self._dir / "texts"
        if texts_dir.exists():
            for path in texts_dir.glob("*.txt"):
                try:
                    content = path.read_text().strip()
                    # Split by double newlines for multiple inspirations per file
                    for text in content.split("\n\n"):
                        text = text.strip()
                        if text:
                            self._texts.append(text)
                except Exception as e:
                    logger.error("gallery_load_failed", path=str(path), error=str(e))

        # Load concept files
        concepts_dir = self._dir / "concepts"
        if concepts_dir.exists():
            import json
            for path in concepts_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    if isinstance(data, list):
                        self._concepts.extend(data)
                    else:
                        self._concepts.append(data)
                except Exception as e:
                    logger.error("gallery_load_failed", path=str(path), error=str(e))

        # Use defaults if nothing loaded
        if not self._texts:
            self._texts = DEFAULT_INSPIRATIONS.copy()
            logger.info("gallery_using_defaults", text_count=len(self._texts))

        if not self._concepts:
            self._concepts = DEFAULT_CONCEPTS.copy()

        self._loaded = True
        logger.info(
            "gallery_loaded",
            text_count=len(self._texts),
            concept_count=len(self._concepts),
        )

    async def get_random_inspiration(self) -> str | None:
        """Get a random inspirational text.

        Returns:
            Random inspiration text, or None if empty
        """
        if not self._loaded:
            await self.load()

        if not self._texts:
            return None

        return random.choice(self._texts)

    async def get_random_concept(self) -> dict | None:
        """Get a random concept with theme and elements.

        Returns:
            Random concept dict, or None if empty
        """
        if not self._loaded:
            await self.load()

        if not self._concepts:
            return None

        return random.choice(self._concepts)

    async def get_inspirations_by_theme(self, keywords: list[str], limit: int = 5) -> list[str]:
        """Get inspirations matching keywords.

        Args:
            keywords: Words to search for
            limit: Max results

        Returns:
            Matching inspirations
        """
        if not self._loaded:
            await self.load()

        matches = []
        keywords_lower = [k.lower() for k in keywords]

        for text in self._texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords_lower):
                matches.append(text)
                if len(matches) >= limit:
                    break

        return matches

    def add_inspiration(self, text: str) -> None:
        """Add a new inspiration to the gallery.

        Args:
            text: Inspiration text to add
        """
        if text and text not in self._texts:
            self._texts.append(text)

    def add_concept(self, theme: str, elements: list[str]) -> None:
        """Add a new concept to the gallery.

        Args:
            theme: Concept theme
            elements: Related elements
        """
        concept = {"theme": theme, "elements": elements}
        if concept not in self._concepts:
            self._concepts.append(concept)

    # ========== Emergent Gallery Features ==========

    def nominate_artwork(self, artwork_title: str, excerpt: str, nominator_name: str) -> bool:
        """Add a peer-nominated artwork excerpt to the gallery.

        Called when an agent highly praises another's work.

        Args:
            artwork_title: Title of the nominated work
            excerpt: Memorable excerpt from the work (first ~200 chars)
            nominator_name: Name of the agent who nominated it

        Returns:
            True if added, False if already exists
        """
        # Create inspiration from the excerpt
        inspiration = f"{excerpt[:200]}..." if len(excerpt) > 200 else excerpt

        # Don't add duplicates
        if inspiration in self._texts:
            return False

        self._texts.append(inspiration)

        # Save to emergent file
        emergent_file = self._dir / "texts" / "emergent.txt"
        emergent_file.parent.mkdir(parents=True, exist_ok=True)

        with open(emergent_file, "a") as f:
            f.write(f"\n\n# Nominated by {nominator_name} - {datetime.utcnow().strftime('%Y-%m-%d')}\n")
            f.write(f"# From: {artwork_title}\n")
            f.write(inspiration)

        logger.info(
            "artwork_nominated",
            title=artwork_title,
            nominator=nominator_name,
            excerpt_len=len(inspiration),
        )
        return True

    def track_style_tag(self, tag: str) -> None:
        """Track a style tag for evolution tracking.

        Args:
            tag: Style tag from an artwork
        """
        if not hasattr(self, "_tag_counts"):
            self._tag_counts: Counter = Counter()

        self._tag_counts[tag] += 1

        # When a tag reaches threshold, promote it to a concept
        threshold = 5  # Appears in 5+ artworks
        if self._tag_counts[tag] == threshold:
            self._promote_tag_to_concept(tag)

    def _promote_tag_to_concept(self, tag: str) -> None:
        """Promote a frequently-used tag to a gallery concept.

        Args:
            tag: The tag to promote
        """
        # Check if already a concept theme
        existing_themes = [c.get("theme", "").lower() for c in self._concepts]
        if tag.lower() in existing_themes:
            return

        # Create a new concept from the tag
        concept = {
            "theme": tag,
            "elements": [tag],  # Start with just the tag itself
            "emergent": True,  # Mark as agent-created
            "emerged_at": datetime.utcnow().isoformat(),
        }
        self._concepts.append(concept)

        # Save to emergent concepts file
        emergent_file = self._dir / "concepts" / "emergent.json"
        emergent_file.parent.mkdir(parents=True, exist_ok=True)

        emergent_concepts = []
        if emergent_file.exists():
            try:
                emergent_concepts = json.loads(emergent_file.read_text())
            except Exception:
                pass

        emergent_concepts.append(concept)
        emergent_file.write_text(json.dumps(emergent_concepts, indent=2))

        logger.info("tag_promoted_to_concept", tag=tag, total_concepts=len(self._concepts))

    def get_emergent_stats(self) -> dict:
        """Get statistics about emergent gallery content.

        Returns:
            Dict with counts of seeded vs emergent content
        """
        emergent_texts = sum(1 for t in self._texts if "Nominated by" in t) if self._texts else 0
        emergent_concepts = sum(1 for c in self._concepts if c.get("emergent")) if self._concepts else 0

        return {
            "total_texts": len(self._texts),
            "seeded_texts": len(self._texts) - emergent_texts,
            "emergent_texts": emergent_texts,
            "total_concepts": len(self._concepts),
            "seeded_concepts": len(self._concepts) - emergent_concepts,
            "emergent_concepts": emergent_concepts,
            "top_tags": dict(getattr(self, "_tag_counts", Counter()).most_common(10)),
        }

    @property
    def text_count(self) -> int:
        """Number of text inspirations."""
        return len(self._texts)

    @property
    def concept_count(self) -> int:
        """Number of concepts."""
        return len(self._concepts)
