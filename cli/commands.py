"""CLI commands for nanobot."""

import argparse
import asyncio
import os
import select
import signal
import sys
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

__version__ = "0.1.4.post4"
__logo__ = "👋🏻"
from config.paths import get_workspace_path
from config.schema import Config
from utils.helpers import sync_workspace_templates

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: readline for history, input() for reading
# ---------------------------------------------------------------------------

_HISTORY_LOADED = False
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_readline() -> None:
    """Set up readline with persistent file history."""
    global _HISTORY_LOADED, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    if _HISTORY_LOADED:
        return

    try:
        import readline
    except ImportError:
        _HISTORY_LOADED = True
        return

    from config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        readline.read_history_file(str(history_file))
    except Exception:
        pass

    import atexit
    atexit.register(readline.write_history_file, str(history_file))
    _HISTORY_LOADED = True


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print("[cyan]CC:[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


def _read_interactive_input() -> str:
    """Read user input using input() with readline support."""
    try:
        console.print("[bold blue]You:[/bold blue]")
        return input("")
    except EOFError as exc:
        raise KeyboardInterrupt from exc


# ============================================================================
# Onboard / Setup
# ============================================================================


def cmd_onboard(args: argparse.Namespace) -> None:
    """Initialize nanobot configuration and workspace."""
    from config.loader import get_config_path, load_config, save_config

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        answer = input("Overwrite? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")

    # Create workspace
    workspace = get_workspace_path()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    sync_workspace_templates(workspace)

    console.print(f"\n{__logo__} CC is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your Gemini API key to [cyan]~/.nanobot/config.json[/cyan]")
    console.print("     Get one at: https://aistudio.google.com/apikey")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")


def _make_provider(config: Config):
    """Create the Gemini LLM provider from config."""
    from providers.gemini_provider import GeminiProvider

    model = config.agents.defaults.model
    p = config.get_provider(model)

    if not p or not p.api_key:
        console.print("[red]Error: No Gemini API key configured.[/red]")
        console.print("Set one in ~/.nanobot/config.json under providers.gemini")
        sys.exit(1)

    return GeminiProvider(
        api_key=p.api_key,
        api_base=config.get_api_base(model),
        retry_config=config.agents.defaults.retry.model_dump(),
        default_model=model,
    )


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            sys.exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


# ============================================================================
# Agent Commands
# ============================================================================


def cmd_agent(args: argparse.Namespace) -> None:
    """Interact with the agent directly."""
    from agent.loop import AgentLoop

    config = _load_runtime_config(args.config, args.workspace)
    sync_workspace_templates(config.workspace_path)

    provider = _make_provider(config)

    agent_loop = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        brave_api_key=config.tools.web.search.api_key or None,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
    )

    render_markdown = not args.no_markdown

    def _thinking_ctx():
        console.print()
        return console.status("[dim]CC is thinking...[/dim]", spinner="dots")

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        console.print(f"  [dim]↳ {content}[/dim]")

    if args.message:
        # Single message mode (Programmatic JSON Output)
        json_traces = []

        async def _json_progress(content: str, *, tool_hint: bool = False) -> None:
            json_traces.append(content)

        async def run_once():
            response = await agent_loop.process_direct(args.message, args.session, on_progress=_json_progress)
            
            import json
            print(json.dumps({
                "response": response or "",
                "trace": json_traces
            }))
            
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode
        console.print()
        _init_readline()

        def _handle_signal(signum, frame):
            _restore_terminal()
            console.print()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGHUP is not available on Windows
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        # Ignore SIGPIPE to prevent silent process termination when writing to closed pipes
        # SIGPIPE is not available on Windows
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive():
            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = _read_interactive_input()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            break

                        with _thinking_ctx():
                            response = await agent_loop.process_direct(
                                command, args.session, on_progress=_cli_progress
                            )

                        _print_agent_response(response, render_markdown=render_markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print()
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print()
                        break
            finally:
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Status Commands
# ============================================================================


def cmd_status(args: argparse.Namespace) -> None:
    """Show nanobot status."""
    from config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} CC Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        console.print(f"Model: {config.agents.defaults.model}")
        has_key = bool(config.providers.gemini.api_key)
        console.print(f"Gemini: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# Argument Parser
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanobot",
        description=f"{__logo__} CC - Personal AI Assistant",
    )
    parser.add_argument("--version", "-v", action="version", version=f"{__logo__} CC v{__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # onboard
    subparsers.add_parser("onboard", help="Initialize nanobot configuration and workspace.")

    # agent
    agent_parser = subparsers.add_parser("agent", help="Interact with the agent directly.")
    agent_parser.add_argument("--message", "-m", default=None, help="Message to send to the agent")
    agent_parser.add_argument("--session", "-s", default="cli:direct", help="Session ID")
    agent_parser.add_argument("--workspace", "-w", default=None, help="Workspace directory")
    agent_parser.add_argument("--config", "-c", default=None, help="Config file path")
    agent_parser.add_argument("--no-markdown", action="store_true", help="Disable Markdown rendering")

    # status
    subparsers.add_parser("status", help="Show nanobot status.")

    return parser


_COMMANDS = {
    "onboard": cmd_onboard,
    "agent": cmd_agent,
    "status": cmd_status,
}


def app(args: list[str] | None = None) -> None:
    """Main entry point for the nanobot CLI."""
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        sys.exit(0)

    handler = _COMMANDS[parsed.command]
    handler(parsed)


if __name__ == "__main__":
    app()
