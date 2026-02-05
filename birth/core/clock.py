"""Simulation time tracking."""

from datetime import datetime, timedelta


class SimulationClock:
    """Tracks simulation time and cycles."""

    def __init__(self):
        self._start_time = datetime.utcnow()
        self._cycle_count = 0
        self._paused = False
        self._pause_start: datetime | None = None
        self._total_pause_duration = timedelta()

    @property
    def start_time(self) -> datetime:
        """When the simulation started."""
        return self._start_time

    @property
    def elapsed(self) -> timedelta:
        """Total elapsed time, excluding pauses."""
        if self._paused and self._pause_start:
            return datetime.utcnow() - self._start_time - self._total_pause_duration - (
                datetime.utcnow() - self._pause_start
            )
        return datetime.utcnow() - self._start_time - self._total_pause_duration

    @property
    def cycle_count(self) -> int:
        """Total number of agent cycles completed."""
        return self._cycle_count

    @property
    def is_paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    def tick(self) -> int:
        """Increment cycle count. Returns new count."""
        self._cycle_count += 1
        return self._cycle_count

    def pause(self) -> None:
        """Pause the simulation clock."""
        if not self._paused:
            self._paused = True
            self._pause_start = datetime.utcnow()

    def resume(self) -> None:
        """Resume the simulation clock."""
        if self._paused and self._pause_start:
            self._total_pause_duration += datetime.utcnow() - self._pause_start
            self._paused = False
            self._pause_start = None

    def reset(self) -> None:
        """Reset the clock to initial state."""
        self._start_time = datetime.utcnow()
        self._cycle_count = 0
        self._paused = False
        self._pause_start = None
        self._total_pause_duration = timedelta()

    def status(self) -> dict:
        """Get clock status as dictionary."""
        return {
            "start_time": self._start_time.isoformat(),
            "elapsed_seconds": self.elapsed.total_seconds(),
            "cycle_count": self._cycle_count,
            "is_paused": self._paused,
        }
