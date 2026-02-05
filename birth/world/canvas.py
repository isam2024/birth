"""Canvas - The simulated world container.

The Canvas is the passive, shared environment where agents exist.
It is their studio, gallery, and commons - the Rules that govern their existence.
"""

from pathlib import Path

from birth.config import Config, get_config
from birth.core.events import EventBus
from birth.integrations.ollama import OllamaClient
from birth.integrations.stable_diffusion import StableDiffusionClient
from birth.observation.logger import get_logger
from birth.storage.database import Database
from birth.storage.repository import Repository
from birth.world.challenges import ChallengeManager
from birth.world.commons import Commons
from birth.world.drops import DropsWatcher
from birth.world.gallery import Gallery

logger = get_logger("birth.canvas")


class Canvas:
    """The simulated world - studio, gallery, and commons.

    The Canvas provides the space but does not judge or direct agents' actions.
    It enforces the simple physics: agents can observe, create, and communicate.
    """

    def __init__(
        self,
        config: Config | None = None,
        database: Database | None = None,
        ollama: OllamaClient | None = None,
        sd_client: StableDiffusionClient | None = None,
        event_bus: EventBus | None = None,
    ):
        self._config = config or get_config()
        self._database = database
        self._ollama = ollama
        self._sd_client = sd_client
        self._event_bus = event_bus or EventBus()

        self._repository: Repository | None = None
        self._gallery: Gallery | None = None
        self._commons: Commons | None = None
        self._drops: DropsWatcher | None = None
        self._challenges: ChallengeManager | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Canvas and all its components."""
        if self._initialized:
            return

        # Initialize database if not provided
        if not self._database:
            from birth.storage.database import get_database
            self._database = await get_database(self._config)

        # Create repository
        self._repository = Repository(self._database)

        # Initialize Gallery
        self._gallery = Gallery(self._config.gallery_dir)
        await self._gallery.load()

        # Initialize Commons
        self._commons = Commons(
            repository=self._repository,
            artworks_dir=self._config.artworks_dir,
            sd_client=self._sd_client,
        )

        # Initialize Drops watcher
        self._drops = DropsWatcher(
            drops_dir=self._config.drops_dir,
            ollama=self._ollama,
            event_bus=self._event_bus,
        )

        # Initialize Challenge manager
        self._challenges = ChallengeManager(event_bus=self._event_bus)

        # Start event bus
        await self._event_bus.start()

        # Start drops watcher if ollama is available
        print(f"\n[DEBUG] Canvas.initialize: ollama={self._ollama is not None}, drops={self._drops is not None}")
        if self._ollama:
            try:
                print("[DEBUG] Starting DropsWatcher...")
                await self._drops.start()
                print("[DEBUG] DropsWatcher started successfully")
            except Exception as e:
                import traceback
                print(f"\n[ERROR] DropsWatcher failed to start: {e}")
                traceback.print_exc()
                logger.warning("drops_watcher_failed", error=str(e))
        else:
            print("\n[WARNING] DropsWatcher not started - no Ollama client")

        self._initialized = True
        logger.info(
            "canvas_initialized",
            gallery_texts=self._gallery.text_count,
            gallery_concepts=self._gallery.concept_count,
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the Canvas."""
        if not self._initialized:
            return

        # Stop drops watcher
        if self._drops:
            await self._drops.stop()

        # Stop event bus
        await self._event_bus.stop()

        # Close database
        if self._database:
            await self._database.close()

        self._initialized = False
        logger.info("canvas_shutdown")

    @property
    def gallery(self) -> Gallery:
        """The Gallery of inspirational assets."""
        if not self._gallery:
            raise RuntimeError("Canvas not initialized")
        return self._gallery

    @property
    def commons(self) -> Commons:
        """The Commons where artworks are shared."""
        if not self._commons:
            raise RuntimeError("Canvas not initialized")
        return self._commons

    @property
    def drops(self) -> DropsWatcher:
        """The Drops watcher for external image input."""
        if not self._drops:
            raise RuntimeError("Canvas not initialized")
        return self._drops

    @property
    def challenges(self) -> ChallengeManager:
        """The Challenge manager for creative prompts."""
        if not self._challenges:
            raise RuntimeError("Canvas not initialized")
        return self._challenges

    @property
    def repository(self) -> Repository:
        """Data repository."""
        if not self._repository:
            raise RuntimeError("Canvas not initialized")
        return self._repository

    @property
    def event_bus(self) -> EventBus:
        """Event bus for inter-agent communication."""
        return self._event_bus

    @property
    def database(self) -> Database:
        """Database instance."""
        if not self._database:
            raise RuntimeError("Canvas not initialized")
        return self._database

    @property
    def is_initialized(self) -> bool:
        """Whether the Canvas is initialized."""
        return self._initialized

    # ========== World Rules ==========

    def get_rules(self) -> dict:
        """Get the rules/physics of this world.

        Returns:
            Dictionary describing world rules
        """
        return {
            "name": "Birth",
            "version": "0.1.0",
            "rules": {
                "creation": "Agents can create artworks freely",
                "observation": "Agents can observe all public creations",
                "communication": "Agents can send messages to each other",
                "sentiment": "Agents develop feelings through interaction",
                "persistence": "All creations persist in the Commons",
                "safety": "Agents cannot be destroyed or harm one another",
            },
            "mediums": ["text", "image", "mixed"],
            "actions": [
                "observe_commons",
                "observe_gallery",
                "create_art",
                "message_agent",
                "critique_art",
                "collaborate",
                "reflect",
                "rest",
            ],
        }

    async def get_world_state(self) -> dict:
        """Get current state of the world.

        Returns:
            Dictionary with world state summary
        """
        if not self._initialized:
            return {"initialized": False}

        # Get active agents
        agents = await self._repository.get_active_agents()

        # Get recent artworks
        artworks = await self._commons.get_recent(10)

        return {
            "initialized": True,
            "active_agents": len(agents),
            "recent_artworks": len(artworks),
            "gallery_items": self._gallery.text_count + self._gallery.concept_count,
            "pending_events": self._event_bus.pending_count,
            "drops_count": len(self._drops.drops) if self._drops else 0,
        }


async def create_canvas(
    config: Config | None = None,
    with_ollama: bool = True,
    with_sd: bool = False,
) -> Canvas:
    """Factory function to create and initialize a Canvas.

    Args:
        config: Configuration to use
        with_ollama: Whether to initialize Ollama client
        with_sd: Whether to initialize Stable Diffusion client

    Returns:
        Initialized Canvas
    """
    config = config or get_config()

    ollama = None
    sd_client = None

    if with_ollama:
        from birth.integrations.ollama import OllamaClient
        ollama = OllamaClient(config.ollama)
        try:
            await ollama.connect()
        except Exception as e:
            logger.warning("ollama_connection_failed", error=str(e))
            ollama = None

    if with_sd:
        from birth.integrations.stable_diffusion import StableDiffusionClient
        sd_client = StableDiffusionClient(config.stable_diffusion)
        try:
            await sd_client.connect()
        except Exception as e:
            logger.warning("sd_connection_failed", error=str(e))
            sd_client = None

    canvas = Canvas(
        config=config,
        ollama=ollama,
        sd_client=sd_client,
    )

    await canvas.initialize()
    return canvas
