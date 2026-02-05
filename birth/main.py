"""Birth: Emergent Digital Creativity Simulation.

Entry point for running the simulation.
"""

import argparse
import asyncio
import sys

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from birth import __version__
from birth.config import get_config, reload_config
from birth.observation.logger import setup_logging, get_logger

console = Console()
logger = get_logger("birth.main")


def create_status_display(engine) -> Table:
    """Create a status display table."""
    table = Table(title="Birth Simulation", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    status = engine.get_status()
    metrics = engine.get_metrics()

    table.add_row("Status", "Running" if status["running"] else "Stopped")
    table.add_row("Paused", "Yes" if status["paused"] else "No")
    table.add_row("Cycles", str(metrics["cycles"]))
    table.add_row("Elapsed", f"{metrics['elapsed_seconds']:.1f}s")
    table.add_row("Agents", str(metrics["agents"]["count"]))
    table.add_row("Avg Mood", f"{metrics['agents']['avg_mood']:.2f}")
    table.add_row("Avg Energy", f"{metrics['agents']['avg_energy']:.2f}")

    return table


async def run_interactive(agent_count: int, challenge: str | None = None) -> None:
    """Run simulation with interactive status display."""
    from birth.core.engine import SimulationEngine
    from birth.integrations.ollama import get_ollama_client, close_ollama_client
    from birth.integrations.stable_diffusion import get_sd_client, close_sd_client
    from birth.world.canvas import create_canvas

    config = get_config()

    console.print(Panel.fit(
        f"[bold blue]Birth[/bold blue] v{__version__}\n"
        f"Emergent Digital Creativity Simulation\n\n"
        f"[dim]Spawning {agent_count} autonomous artist agents...[/dim]",
        border_style="blue",
    ))

    try:
        # Initialize components
        console.print("[dim]Connecting to Ollama...[/dim]")
        ollama = await get_ollama_client()

        # Try SD but don't fail if unavailable
        sd_client = None
        try:
            console.print("[dim]Connecting to Stable Diffusion...[/dim]")
            sd_client = await get_sd_client()
            console.print("[green]Stable Diffusion connected - multi-modal art enabled[/green]")
        except Exception as e:
            console.print(f"[yellow]Stable Diffusion unavailable - text-only mode[/yellow]")

        console.print("[dim]Initializing canvas...[/dim]")
        canvas = await create_canvas(config, with_ollama=False, with_sd=False)
        # Manually set the clients
        canvas._ollama = ollama
        canvas._sd_client = sd_client

        # Start drops watcher now that ollama is set
        if ollama and canvas._drops:
            canvas._drops.set_clients(ollama, canvas._event_bus)
            await canvas._drops.start()

        # Create engine
        engine = SimulationEngine(canvas, ollama, config)

        # Start simulation
        console.print("[dim]Starting simulation...[/dim]")
        await engine.start(agent_count)

        console.print(f"\n[bold green]Simulation running with {engine.agent_count} agents[/bold green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        # Display agent names
        console.print("[bold]Artists in the colony:[/bold]")
        for agent in engine.agents:
            console.print(f"  • {agent.name}")
        console.print()

        # Issue challenge if provided
        if challenge:
            console.print(f"\n[bold yellow]CREATIVE CHALLENGE ISSUED:[/bold yellow]")
            console.print(f"[yellow]\"{challenge}\"[/yellow]")
            console.print("[dim]All agents will respond with their interpretation...[/dim]\n")
            await canvas.challenges.issue_challenge(challenge)

        # Run with live status updates
        snapshot_interval = 50  # Take snapshot every N cycles
        last_snapshot = 0

        try:
            while engine.is_running:
                await asyncio.sleep(5)

                # Print periodic status
                metrics = engine.get_metrics()
                console.print(
                    f"[dim]Cycle {metrics['cycles']} | "
                    f"Mood: {metrics['agents']['avg_mood']:.2f} | "
                    f"Energy: {metrics['agents']['avg_energy']:.2f}[/dim]"
                )

                # Take periodic snapshots
                if metrics['cycles'] - last_snapshot >= snapshot_interval:
                    await engine.take_snapshot()
                    last_snapshot = metrics['cycles']
                    console.print(f"[magenta]Snapshot saved at cycle {metrics['cycles']}[/magenta]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping simulation...[/yellow]")
            await engine.stop("user_interrupt")

        # Final status
        final_metrics = engine.get_metrics()
        config = get_config()
        console.print(Panel.fit(
            f"[bold]Simulation Complete[/bold]\n\n"
            f"Total cycles: {final_metrics['cycles']}\n"
            f"Duration: {final_metrics['elapsed_seconds']:.1f} seconds\n"
            f"Agents: {final_metrics['agents']['count']}\n\n"
            f"[dim]Snapshots saved to: {config.output_dir / 'snapshots'}[/dim]",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("simulation_error")
        raise

    finally:
        # Cleanup
        await close_ollama_client()
        await close_sd_client()
        if 'canvas' in locals():
            await canvas.shutdown()


async def run_headless(agent_count: int, duration: float | None) -> None:
    """Run simulation without interactive display."""
    from birth.core.engine import run_simulation

    config = get_config()

    logger.info(
        "starting_headless",
        agent_count=agent_count,
        duration=duration,
    )

    engine = await run_simulation(
        agent_count=agent_count,
        duration=duration,
        config=config,
    )

    logger.info(
        "simulation_complete",
        cycles=engine.get_metrics()["cycles"],
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Birth: Emergent Digital Creativity Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  birth                     Run with default settings (20 agents)
  birth -n 5                Run with 5 agents
  birth -n 10 --headless    Run headless for automation
  birth --duration 300      Run for 5 minutes then stop
        """,
    )

    parser.add_argument(
        "-n", "--agents",
        type=int,
        default=None,
        help="Number of agents to spawn (default: from config)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (default: run forever)",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive display",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"Birth {__version__}",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (all messages)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - only show creations, reflections, and errors",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Set log level (default: INFO)",
    )

    parser.add_argument(
        "--challenge",
        type=str,
        default=None,
        help="Issue a creative challenge/theme for all agents to respond to",
    )

    args = parser.parse_args()

    # Load config first so we can modify it
    config = reload_config()

    # Apply command line overrides
    if args.debug:
        config.log_level = "DEBUG"
        config.agent_log_level = "DEBUG"
    elif args.log_level:
        config.log_level = args.log_level
        config.agent_log_level = args.log_level

    if args.quiet:
        config.quiet_mode = True

    # Setup logging (uses config settings)
    setup_logging()

    # Determine agent count
    agent_count = args.agents or config.simulation.initial_agent_count

    # Run simulation
    try:
        if args.headless:
            asyncio.run(run_headless(agent_count, args.duration))
        else:
            asyncio.run(run_interactive(agent_count, challenge=args.challenge))
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130

    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.exception("fatal_error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
