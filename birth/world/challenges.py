"""Creative challenges - prompts that inspire all agents to create."""

import asyncio
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Prompt

from birth.config import get_config
from birth.core.events import Event, EventType
from birth.observation.logger import get_logger

if TYPE_CHECKING:
    from birth.core.events import EventBus

logger = get_logger("birth.challenges")
console = Console()


@dataclass
class Challenge:
    """A creative challenge/prompt for agents."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""  # The creative prompt/theme
    image_path: str | None = None  # Optional reference image
    created_at: datetime = field(default_factory=datetime.utcnow)
    responses: set[str] = field(default_factory=set)  # Agent IDs who have responded

    def has_responded(self, agent_id: str) -> bool:
        return agent_id in self.responses

    def mark_responded(self, agent_id: str) -> None:
        self.responses.add(agent_id)


class ChallengeManager:
    """Manages creative challenges for the colony.

    Challenges can be issued:
    1. Via CLI at startup: --challenge "theme"
    2. Interactively by pressing 'c' during simulation
    """

    def __init__(self, event_bus: "EventBus | None" = None):
        self._event_bus = event_bus
        self._active_challenge: Challenge | None = None
        self._challenge_history: list[Challenge] = []
        config = get_config()
        self._challenges_dir = config.data_dir / "challenges"
        self._challenges_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_challenge(self) -> Challenge | None:
        return self._active_challenge

    @property
    def pending_responses(self) -> int:
        """Number of agents who haven't responded yet."""
        if not self._active_challenge:
            return 0
        # This would need agent count passed in - for now just return if active
        return 1 if self._active_challenge else 0

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    async def issue_challenge(
        self,
        prompt: str,
        image_path: str | None = None,
    ) -> Challenge:
        """Issue a new creative challenge to all agents.

        Args:
            prompt: The creative prompt/theme
            image_path: Optional path to a reference image

        Returns:
            The created Challenge
        """
        # Archive previous challenge if any
        if self._active_challenge:
            self._challenge_history.append(self._active_challenge)

        challenge = Challenge(
            prompt=prompt,
            image_path=image_path,
        )
        self._active_challenge = challenge

        # Display to console
        console.print(f"[bold yellow]CREATIVE CHALLENGE ISSUED:[/bold yellow]")
        console.print(f"[yellow]\"{prompt}\"[/yellow]")
        console.print("[dim]All agents will respond with their interpretation...[/dim]\n")

        logger.info(
            "challenge_issued",
            challenge_id=challenge.id,
            prompt=prompt[:100],
            has_image=image_path is not None,
        )

        # Broadcast to all agents
        if self._event_bus:
            await self._event_bus.publish(Event(
                type=EventType.CHALLENGE_ISSUED,
                source_agent_id=None,
                data={
                    "challenge_id": challenge.id,
                    "prompt": prompt,
                    "image_path": image_path,
                },
            ))

        return challenge

    def clear_challenge(self) -> None:
        """Clear the active challenge."""
        if self._active_challenge:
            self._challenge_history.append(self._active_challenge)
            self._active_challenge = None
            logger.info("challenge_cleared")

    def get_challenge_for_agent(self, agent_id: str) -> Challenge | None:
        """Get the active challenge if agent hasn't responded yet."""
        if self._active_challenge and not self._active_challenge.has_responded(agent_id):
            return self._active_challenge
        return None

    def mark_agent_responded(self, agent_id: str) -> None:
        """Mark that an agent has responded to the current challenge."""
        if self._active_challenge:
            self._active_challenge.mark_responded(agent_id)
            logger.debug(
                "challenge_response",
                agent_id=agent_id,
                challenge_id=self._active_challenge.id,
            )
