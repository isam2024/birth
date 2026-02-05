"""Structured logging for Birth simulation."""

import sys
from pathlib import Path

import structlog
from rich.console import Console

from birth.config import get_config


def setup_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging for the simulation."""
    config = get_config()

    # Ensure logs directory exists
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if sys.stdout.isatty():
        # Rich console output for interactive use
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )
    else:
        # JSON output for file/programmatic use
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("birth")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger("birth")


class AgentLogger:
    """Logger wrapper for agent-specific logging."""

    def __init__(self, agent_id: str, agent_name: str):
        self._logger = get_logger("birth.agent")
        self._agent_id = agent_id
        self._agent_name = agent_name

    def _bind(self) -> structlog.stdlib.BoundLogger:
        return self._logger.bind(
            agent_id=self._agent_id,
            agent_name=self._agent_name,
        )

    def perception(self, content: str, **kwargs) -> None:
        """Log a perception event."""
        self._bind().info("perception", content=content, **kwargs)

    def thought(self, content: str, **kwargs) -> None:
        """Log a thought/internal process."""
        self._bind().debug("thought", content=content, **kwargs)

    def action(self, action_type: str, **kwargs) -> None:
        """Log an action taken."""
        self._bind().info("action", action_type=action_type, **kwargs)

    def creation(self, artwork_id: str, title: str, medium: str, **kwargs) -> None:
        """Log artwork creation."""
        self._bind().info(
            "creation",
            artwork_id=artwork_id,
            title=title,
            medium=medium,
            **kwargs,
        )

    def sentiment_change(
        self, target_type: str, target_id: str, old_score: float, new_score: float
    ) -> None:
        """Log a sentiment change."""
        self._bind().debug(
            "sentiment_change",
            target_type=target_type,
            target_id=target_id,
            old_score=old_score,
            new_score=new_score,
            delta=new_score - old_score,
        )

    def reflection(self, insight: str, memory_count: int) -> None:
        """Log a reflection event."""
        self._bind().info(
            "reflection",
            insight=insight,
            memory_count=memory_count,
        )

    def interaction(
        self, interaction_type: str, target_agent_id: str, **kwargs
    ) -> None:
        """Log an interaction with another agent."""
        self._bind().info(
            "interaction",
            interaction_type=interaction_type,
            target_agent_id=target_agent_id,
            **kwargs,
        )

    def error(self, message: str, **kwargs) -> None:
        """Log an error."""
        self._bind().error(message, **kwargs)


class DropsLogger:
    """Logger for drop events - highlighted in red for visibility."""

    def __init__(self):
        self._console = Console()
        self._logger = get_logger("birth.drops")

    def _red(self, message: str) -> None:
        """Print a message in red."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._console.print(f"[bold red][{timestamp}] [DROP] {message}[/bold red]")

    def processing(self, filename: str) -> None:
        """Log that a drop is being processed."""
        self._red(f"Processing image: {filename}")

    def processed(self, drop_id: str, filename: str, description_preview: str = "") -> None:
        """Log that a drop was successfully processed."""
        preview = f" - {description_preview[:80]}..." if description_preview else ""
        self._red(f"Processed: {filename} (id: {drop_id}){preview}")

    def broadcast(self, drop_id: str) -> None:
        """Log that a drop was broadcast to agents."""
        self._red(f"Broadcast to all agents: {drop_id}")

    def viewed(self, agent_name: str, filename: str) -> None:
        """Log that an agent viewed a drop."""
        self._red(f"Agent '{agent_name}' viewed: {filename}")

    def opinion(self, agent_name: str, filename: str, score: float, reason: str = "") -> None:
        """Log an agent's opinion on a drop."""
        sentiment = "loves" if score > 0.5 else "likes" if score > 0 else "dislikes" if score > -0.5 else "hates"
        reason_text = f" - {reason}" if reason else ""
        self._red(f"Agent '{agent_name}' {sentiment} '{filename}' ({score:+.2f}){reason_text}")

    def error(self, message: str, **kwargs) -> None:
        """Log a drop-related error."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._red(f"ERROR: {message} {extra}")


class SimulationLogger:
    """Logger for simulation-wide events."""

    def __init__(self):
        self._logger = get_logger("birth.simulation")

    def started(self, agent_count: int, **kwargs) -> None:
        """Log simulation start."""
        self._logger.info("simulation_started", agent_count=agent_count, **kwargs)

    def stopped(self, reason: str, **kwargs) -> None:
        """Log simulation stop."""
        self._logger.info("simulation_stopped", reason=reason, **kwargs)

    def paused(self) -> None:
        """Log simulation pause."""
        self._logger.info("simulation_paused")

    def resumed(self) -> None:
        """Log simulation resume."""
        self._logger.info("simulation_resumed")

    def cycle_complete(self, cycle_number: int, active_agents: int, **kwargs) -> None:
        """Log cycle completion."""
        self._logger.debug(
            "cycle_complete",
            cycle_number=cycle_number,
            active_agents=active_agents,
            **kwargs,
        )

    def agent_spawned(self, agent_id: str, agent_name: str) -> None:
        """Log agent spawn."""
        self._logger.info("agent_spawned", agent_id=agent_id, agent_name=agent_name)

    def agent_deactivated(self, agent_id: str, reason: str) -> None:
        """Log agent deactivation."""
        self._logger.info("agent_deactivated", agent_id=agent_id, reason=reason)

    def error(self, message: str, **kwargs) -> None:
        """Log a simulation error."""
        self._logger.error(message, **kwargs)

    def metrics(self, **kwargs) -> None:
        """Log simulation metrics."""
        self._logger.info("metrics", **kwargs)
