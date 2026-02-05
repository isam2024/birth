"""Gallery - Repository of inspirational assets.

The Gallery is the "outside world" that agents can observe for inspiration.
It contains pre-seeded texts, concepts, and ideas.
"""

import random
from pathlib import Path

from birth.config import get_config
from birth.observation.logger import get_logger

logger = get_logger("birth.gallery")


# DISABLED: Pre-seeded content commented out to test spontaneous emergence
# Uncomment to re-enable gallery influence
#
# DEFAULT_INSPIRATIONS = [
#     # Nature
#     "The way morning light fractures through mist over a still lake.",
#     "A single leaf falling in an empty forest - does it make art?",
#     "The geometry of honeycombs: nature's answer to efficiency.",
#     "Storm clouds gathering at dusk, violet and amber warnings.",
#     "The patience of mountains, wearing down grain by grain.",
#
#     # Emotion
#     "The weight of an unspoken word between old friends.",
#     "Joy so sharp it feels like grief in reverse.",
#     "The hollow echo of footsteps in an empty house.",
#     "That moment of suspension before tears arrive.",
#     "Love that outlives the lovers themselves.",
#
#     # Philosophy
#     "If you could paint silence, what color would it be?",
#     "The space between heartbeats contains multitudes.",
#     "Every ending is a door pretending to be a wall.",
#     "We are all translations of something that has no original.",
#     "The universe experiencing itself through temporary eyes.",
#
#     # Art
#     "A canvas that refuses to hold the paint.",
#     "The conversation between negative space and form.",
#     "When the frame becomes part of the work.",
#     "Art that exists only in the moment of its destruction.",
#     "The ghost of the artist's hand in every brushstroke.",
#
#     # Memory
#     "The smell of a place you've never been but somehow remember.",
#     "Photographs that remember more than you do.",
#     "The way certain songs are time machines.",
#     "Dreams that feel more real than waking.",
#     "Nostalgia for futures that never happened.",
#
#     # Paradox
#     "The more you look, the less you see.",
#     "Perfect imperfection.",
#     "The loudness of absolute silence.",
#     "Finding home in displacement.",
#     "The weight of emptiness.",
# ]
#
# DEFAULT_CONCEPTS = [
#     {"theme": "transience", "elements": ["water", "light", "shadow", "time"]},
#     {"theme": "connection", "elements": ["threads", "bridges", "mirrors", "echoes"]},
#     {"theme": "solitude", "elements": ["empty rooms", "single objects", "vast spaces"]},
#     {"theme": "transformation", "elements": ["metamorphosis", "decay", "growth", "fire"]},
#     {"theme": "duality", "elements": ["light/dark", "presence/absence", "sound/silence"]},
#     {"theme": "memory", "elements": ["fragments", "fading", "layers", "traces"]},
#     {"theme": "chaos", "elements": ["entropy", "storms", "fractals", "unraveling"]},
#     {"theme": "order", "elements": ["geometry", "patterns", "symmetry", "grids"]},
# ]

# Empty defaults - agents create from their own emergence
DEFAULT_INSPIRATIONS: list[str] = []
DEFAULT_CONCEPTS: list[dict] = []


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

    @property
    def text_count(self) -> int:
        """Number of text inspirations."""
        return len(self._texts)

    @property
    def concept_count(self) -> int:
        """Number of concepts."""
        return len(self._concepts)
