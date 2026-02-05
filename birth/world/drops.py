"""Drops - External input system for the simulation.

Allows humans to "drop" images into the simulation for agents to perceive.
Images are analyzed by a vision model and broadcast to all agents.
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from birth.config import get_config
from birth.core.events import Event, EventBus, EventType
from birth.integrations.ollama import OllamaClient
from birth.observation.logger import DropsLogger, get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("birth.drops")
drops_logger = DropsLogger()

# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@dataclass
class Drop:
    """A dropped image and its description."""

    id: str
    filename: str
    path: Path
    description: str
    dropped_at: datetime = field(default_factory=datetime.utcnow)
    viewed_by: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": str(self.path),
            "description": self.description,
            "dropped_at": self.dropped_at.isoformat(),
            "viewed_by": list(self.viewed_by),
        }


class DropsWatcher:
    """Watches a directory for new image drops and broadcasts them to agents."""

    def __init__(
        self,
        drops_dir: Path | None = None,
        ollama: OllamaClient | None = None,
        event_bus: EventBus | None = None,
        poll_interval: float = 2.0,
    ):
        config = get_config()
        self._drops_dir = drops_dir or config.drops_dir
        self._ollama = ollama
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._vision_model = config.ollama.vision_model

        # Track processed files by hash to avoid re-processing
        self._processed_hashes: set[str] = set()

        # Store drops for agent access
        self._drops: dict[str, Drop] = {}

        # State
        self._running = False
        self._watcher_task: asyncio.Task | None = None

        # Ensure drops directory exists
        self._drops_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir = self._drops_dir / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def drops_dir(self) -> Path:
        """Path to drops directory."""
        return self._drops_dir

    @property
    def drops(self) -> list[Drop]:
        """All processed drops."""
        return list(self._drops.values())

    @property
    def recent_drops(self) -> list[Drop]:
        """Recent drops (last 10)."""
        sorted_drops = sorted(
            self._drops.values(),
            key=lambda d: d.dropped_at,
            reverse=True,
        )
        return sorted_drops[:10]

    def get_drop(self, drop_id: str) -> Drop | None:
        """Get a specific drop by ID."""
        return self._drops.get(drop_id)

    def set_clients(self, ollama: OllamaClient, event_bus: EventBus) -> None:
        """Set required clients after initialization."""
        self._ollama = ollama
        self._event_bus = event_bus

    async def start(self) -> None:
        """Start watching for drops."""
        if self._running:
            return

        if not self._ollama or not self._event_bus:
            raise RuntimeError("DropsWatcher requires ollama and event_bus clients")

        self._running = True

        # Process any existing images first
        await self._scan_existing()

        # Start watcher
        self._watcher_task = asyncio.create_task(self._watch_loop())
        drops_logger._red(f"Watcher STARTED - monitoring: {self._drops_dir}")

    async def stop(self) -> None:
        """Stop watching for drops."""
        self._running = False
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None
        logger.info("drops_watcher_stopped")

    async def _scan_existing(self) -> None:
        """Scan for existing images in drops directory."""
        for path in self._drops_dir.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
                await self._process_image(path)

    async def _watch_loop(self) -> None:
        """Main watch loop - polls for new files."""
        while self._running:
            try:
                await self._check_for_new_drops()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("drops_watch_error", error=str(e))
                await asyncio.sleep(self._poll_interval)

    async def _check_for_new_drops(self) -> None:
        """Check for new image files."""
        for path in self._drops_dir.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            # Check if already processed (by file hash)
            file_hash = self._hash_file(path)
            if file_hash in self._processed_hashes:
                continue

            # New image found
            await self._process_image(path, file_hash)

    def _hash_file(self, path: Path) -> str:
        """Get hash of file for deduplication."""
        hasher = hashlib.md5()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()

    async def _process_image(self, path: Path, file_hash: str | None = None) -> None:
        """Process a new image drop."""
        if file_hash is None:
            file_hash = self._hash_file(path)

        drops_logger.processing(path.name)

        try:
            # Use vision model to describe the image
            description = await self._ollama.describe_image(
                image=path,
                model=self._vision_model,
                style="artistic",
            )

            # Create drop record
            drop = Drop(
                id=file_hash[:12],
                filename=path.name,
                path=path,
                description=description,
            )

            self._drops[drop.id] = drop
            self._processed_hashes.add(file_hash)

            drops_logger.processed(drop.id, path.name, description)

            # Broadcast to all agents
            await self._broadcast_drop(drop)

            # Move file to processed folder
            new_path = self._processed_dir / path.name
            # Handle duplicates by adding hash suffix
            if new_path.exists():
                new_path = self._processed_dir / f"{path.stem}_{file_hash[:8]}{path.suffix}"
            path.rename(new_path)
            drop.path = new_path
            drops_logger._red(f"Moved to processed: {path.name}")

        except Exception as e:
            drops_logger.error("Processing failed", filename=path.name, error=str(e))

    async def _broadcast_drop(self, drop: Drop) -> None:
        """Broadcast a drop event to all agents."""
        event = Event(
            type=EventType.IMAGE_DROPPED,
            source_agent_id=None,  # External source
            data={
                "drop_id": drop.id,
                "filename": drop.filename,
                "description": drop.description,
                "path": str(drop.path),
            },
        )

        await self._event_bus.publish(event)
        drops_logger.broadcast(drop.id)

    async def drop_image(self, image_path: Path | str) -> Drop | None:
        """Manually drop an image into the simulation.

        Args:
            image_path: Path to the image file

        Returns:
            The created Drop, or None on failure
        """
        path = Path(image_path)
        if not path.exists():
            logger.error("drop_file_not_found", path=str(path))
            return None

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.error("drop_unsupported_format", format=path.suffix)
            return None

        file_hash = self._hash_file(path)
        if file_hash in self._processed_hashes:
            # Return existing drop
            for drop in self._drops.values():
                if drop.filename == path.name:
                    return drop
            return None

        await self._process_image(path, file_hash)
        return self._drops.get(file_hash[:12])

    def format_drop_for_perception(self, drop: Drop) -> str:
        """Format a drop for agent perception.

        Args:
            drop: The drop to format

        Returns:
            Formatted perception string
        """
        return (
            f"[EXTERNAL IMAGE: {drop.filename}]\n"
            f"A new image has been shared with the colony:\n\n"
            f"{drop.description}"
        )
