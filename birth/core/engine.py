"""Simulation Engine - The orchestrator of autonomous life.

This engine manages the async lifecycle of all agents in the simulation,
coordinating their independent existence while maintaining system stability.
"""

import asyncio
import signal
from datetime import datetime
from typing import Callable

from birth.agents.agent import AutonomousAgent
from birth.agents.persona import load_or_generate_personas
from birth.config import Config, get_config
from birth.core.clock import SimulationClock
from birth.core.events import Event, EventBus, EventType
from birth.integrations.ollama import OllamaClient
from birth.observation.logger import SimulationLogger, get_logger
from birth.observation.snapshots import SnapshotManager
from birth.storage.models import Agent
from birth.world.canvas import Canvas

logger = get_logger("birth.engine")


class SimulationEngine:
    """Main simulation engine coordinating all agents.

    The engine:
    - Spawns and manages agent lifecycles
    - Handles graceful startup and shutdown
    - Tracks simulation metrics
    - Provides pause/resume functionality
    """

    def __init__(
        self,
        canvas: Canvas,
        ollama: OllamaClient,
        config: Config | None = None,
    ):
        self._canvas = canvas
        self._ollama = ollama
        self._config = config or get_config()

        self._clock = SimulationClock()
        self._sim_logger = SimulationLogger()

        # Agent management
        self._agents: dict[str, AutonomousAgent] = {}
        self._agent_tasks: dict[str, asyncio.Task] = {}

        # State
        self._running = False
        self._paused = False
        self._shutdown_requested = False

        # Snapshots
        self._snapshot_manager = SnapshotManager()
        self._snapshot_interval = 50  # Take snapshot every N cycles
        self._last_snapshot_cycle = 0

        # Callbacks
        self._on_agent_created: list[Callable[[AutonomousAgent], None]] = []
        self._on_cycle_complete: list[Callable[[int], None]] = []

    @property
    def clock(self) -> SimulationClock:
        """Simulation clock."""
        return self._clock

    @property
    def is_running(self) -> bool:
        """Whether simulation is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    @property
    def agent_count(self) -> int:
        """Number of active agents."""
        return len(self._agents)

    @property
    def agents(self) -> list[AutonomousAgent]:
        """List of all active agents."""
        return list(self._agents.values())

    # ========== Lifecycle ==========

    async def start(self, agent_count: int | None = None) -> None:
        """Start the simulation.

        Args:
            agent_count: Number of agents to spawn (default from config)
        """
        if self._running:
            logger.warning("simulation_already_running")
            return

        agent_count = agent_count or self._config.simulation.initial_agent_count

        logger.info("simulation_starting", agent_count=agent_count)

        # Reset state
        self._running = True
        self._paused = False
        self._shutdown_requested = False
        self._clock.reset()

        # Publish start event
        await self._canvas.event_bus.publish(Event(
            type=EventType.SIMULATION_STARTED,
            source_agent_id=None,
            data={"agent_count": agent_count},
        ))

        # Load or generate personas
        personas = await load_or_generate_personas(agent_count, self._ollama)

        # Spawn agents
        for persona in personas:
            await self._spawn_agent(persona)

        self._sim_logger.started(len(self._agents))
        logger.info("simulation_started", agents=len(self._agents))

    async def stop(self, reason: str = "user_requested") -> None:
        """Stop the simulation gracefully.

        Args:
            reason: Reason for stopping
        """
        if not self._running:
            return

        logger.info("simulation_stopping", reason=reason)
        self._shutdown_requested = True
        self._running = False

        # Stop all agents
        for agent in self._agents.values():
            agent.stop()

        # Cancel all agent tasks
        for task in self._agent_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)

        self._agent_tasks.clear()

        # Publish stop event
        await self._canvas.event_bus.publish(Event(
            type=EventType.SIMULATION_STOPPED,
            source_agent_id=None,
            data={"reason": reason},
        ))

        # Take final snapshot
        await self.take_snapshot()
        self._snapshot_manager.save_evolution_summary()

        self._sim_logger.stopped(reason)
        logger.info("simulation_stopped", reason=reason, cycles=self._clock.cycle_count)

    async def take_snapshot(self) -> None:
        """Take a snapshot of all agents' current state."""
        cycle = self._clock.cycle_count
        await self._snapshot_manager.take_snapshot(
            list(self._agents.values()),
            cycle,
            repository=self._canvas.repository,
        )
        self._last_snapshot_cycle = cycle
        logger.info("snapshot_taken", cycle=cycle, agents=len(self._agents))

    def pause(self) -> None:
        """Pause the simulation."""
        if self._running and not self._paused:
            self._paused = True
            self._clock.pause()
            self._sim_logger.paused()
            logger.info("simulation_paused")

    def resume(self) -> None:
        """Resume the simulation."""
        if self._running and self._paused:
            self._paused = False
            self._clock.resume()
            self._sim_logger.resumed()
            logger.info("simulation_resumed")

    # ========== Agent Management ==========

    async def _spawn_agent(self, agent_data: Agent) -> AutonomousAgent:
        """Spawn a new agent into the simulation.

        Args:
            agent_data: Agent persona data

        Returns:
            The spawned agent
        """
        # Save agent to database if new
        existing = await self._canvas.repository.get_agent(agent_data.id)
        if not existing:
            await self._canvas.repository.create_agent(agent_data)

        # Create autonomous agent
        agent = AutonomousAgent(
            agent_data=agent_data,
            repository=self._canvas.repository,
            ollama=self._ollama,
            event_bus=self._canvas.event_bus,
            canvas=self._canvas,
        )

        self._agents[agent.id] = agent

        # Start agent lifecycle as background task
        task = asyncio.create_task(self._run_agent(agent))
        self._agent_tasks[agent.id] = task

        self._sim_logger.agent_spawned(agent.id, agent.name)

        # Publish event
        await self._canvas.event_bus.publish(Event(
            type=EventType.AGENT_JOINED,
            source_agent_id=agent.id,
            data={"name": agent.name},
        ))

        # Callbacks
        for callback in self._on_agent_created:
            callback(agent)

        return agent

    async def _run_agent(self, agent: AutonomousAgent) -> None:
        """Run an agent's lifecycle with pause support.

        Args:
            agent: Agent to run
        """
        try:
            while self._running and agent.is_alive:
                # Check for pause
                while self._paused and self._running:
                    await asyncio.sleep(0.5)

                if not self._running:
                    break

                # Run one cycle
                try:
                    # Agent's internal live() handles the full cycle
                    # We break it down here for better control
                    await agent._perceive()

                    if self._paused:
                        continue

                    intention = await agent._reflect([])
                    action = await agent._decide(intention)
                    result = await agent._act(action, intention)
                    await agent._react(result)

                    # Tick the clock
                    self._clock.tick()

                    # Rest period
                    rest = agent._get_rest_period()
                    await asyncio.sleep(rest)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        "agent_cycle_error",
                        agent_id=agent.id,
                        error=str(e),
                    )
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.debug("agent_cancelled", agent_id=agent.id)
        finally:
            agent.stop()

    async def deactivate_agent(self, agent_id: str, reason: str = "deactivated") -> None:
        """Deactivate an agent.

        Args:
            agent_id: Agent to deactivate
            reason: Reason for deactivation
        """
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.stop()

            if agent_id in self._agent_tasks:
                self._agent_tasks[agent_id].cancel()
                try:
                    await self._agent_tasks[agent_id]
                except asyncio.CancelledError:
                    pass
                del self._agent_tasks[agent_id]

            del self._agents[agent_id]

            # Update database
            await self._canvas.repository.deactivate_agent(agent_id)

            self._sim_logger.agent_deactivated(agent_id, reason)

            await self._canvas.event_bus.publish(Event(
                type=EventType.AGENT_LEFT,
                source_agent_id=agent_id,
                data={"reason": reason},
            ))

    def get_agent(self, agent_id: str) -> AutonomousAgent | None:
        """Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent or None
        """
        return self._agents.get(agent_id)

    # ========== Metrics ==========

    def get_status(self) -> dict:
        """Get current simulation status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "paused": self._paused,
            "clock": self._clock.status(),
            "agents": {
                "total": len(self._agents),
                "names": [a.name for a in self._agents.values()],
            },
            "events_pending": self._canvas.event_bus.pending_count,
        }

    def get_metrics(self) -> dict:
        """Get simulation metrics.

        Returns:
            Metrics dictionary
        """
        agent_moods = [a.mood for a in self._agents.values()]
        agent_energies = [a.energy for a in self._agents.values()]

        return {
            "cycles": self._clock.cycle_count,
            "elapsed_seconds": self._clock.elapsed.total_seconds(),
            "agents": {
                "count": len(self._agents),
                "avg_mood": sum(agent_moods) / len(agent_moods) if agent_moods else 0,
                "avg_energy": sum(agent_energies) / len(agent_energies) if agent_energies else 0,
            },
        }

    # ========== Callbacks ==========

    def on_agent_created(self, callback: Callable[[AutonomousAgent], None]) -> None:
        """Register callback for agent creation."""
        self._on_agent_created.append(callback)

    def on_cycle_complete(self, callback: Callable[[int], None]) -> None:
        """Register callback for cycle completion."""
        self._on_cycle_complete.append(callback)


async def run_simulation(
    agent_count: int | None = None,
    duration: float | None = None,
    config: Config | None = None,
) -> SimulationEngine:
    """Convenience function to run a simulation.

    Args:
        agent_count: Number of agents (default from config)
        duration: Run duration in seconds (None = run forever)
        config: Configuration to use

    Returns:
        The simulation engine after completion
    """
    from birth.world.canvas import create_canvas
    from birth.integrations.ollama import get_ollama_client

    config = config or get_config()

    # Create canvas
    canvas = await create_canvas(config, with_ollama=True, with_sd=True)

    # Get Ollama client
    ollama = await get_ollama_client()

    # Create engine
    engine = SimulationEngine(canvas, ollama, config)

    # Setup signal handlers
    def handle_signal(sig, frame):
        logger.info("signal_received", signal=sig)
        asyncio.create_task(engine.stop("signal"))

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start simulation
    await engine.start(agent_count)

    # Run for duration or forever
    if duration:
        await asyncio.sleep(duration)
        await engine.stop("duration_complete")
    else:
        # Run until stopped
        while engine.is_running:
            await asyncio.sleep(1)

    # Cleanup
    await canvas.shutdown()

    return engine
