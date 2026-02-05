"""Event bus for inter-agent communication."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine


class EventType(str, Enum):
    """Types of events in the simulation."""

    # Artwork events
    ARTWORK_CREATED = "artwork_created"
    ARTWORK_VIEWED = "artwork_viewed"

    # Agent events
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"

    # Interaction events
    MESSAGE_SENT = "message_sent"
    CRITIQUE_POSTED = "critique_posted"
    COLLABORATION_PROPOSED = "collaboration_proposed"
    COLLABORATION_ACCEPTED = "collaboration_accepted"

    # Reflection events
    REFLECTION_GENERATED = "reflection_generated"

    # External input events (drops)
    IMAGE_DROPPED = "image_dropped"
    TEXT_DROPPED = "text_dropped"

    # Challenge events
    CHALLENGE_ISSUED = "challenge_issued"
    CHALLENGE_RESPONSE = "challenge_response"

    # System events
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUMED = "simulation_resumed"
    SIMULATION_STOPPED = "simulation_stopped"


@dataclass
class Event:
    """An event in the simulation."""

    type: EventType
    source_agent_id: str | None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return f"Event({self.type.value}, source={self.source_agent_id})"


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async event bus for decoupled communication."""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._running = False
        self._processor_task: asyncio.Task | None = None

    def subscribe(
        self, event_type: EventType | None, handler: EventHandler
    ) -> None:
        """Subscribe to events of a specific type, or all events if type is None."""
        if event_type is None:
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self, event_type: EventType | None, handler: EventHandler
    ) -> None:
        """Unsubscribe from events."""
        if event_type is None:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> bool:
        """Publish an event without waiting. Returns False if queue is full."""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Call type-specific handlers
            handlers = self._handlers.get(event.type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    # Log but don't crash on handler errors
                    print(f"Error in event handler for {event.type}: {e}")

            # Call global handlers
            for handler in self._global_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"Error in global event handler: {e}")

            self._queue.task_done()

    async def start(self) -> None:
        """Start processing events."""
        if not self._running:
            self._running = True
            self._processor_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop processing events."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def drain(self) -> None:
        """Wait for all events to be processed."""
        await self._queue.join()

    @property
    def pending_count(self) -> int:
        """Number of events waiting to be processed."""
        return self._queue.qsize()
