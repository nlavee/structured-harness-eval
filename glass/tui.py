"""GLASS Terminal User Interface — rich-based console with dual logging."""

import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ──────────────────────────────────────────────────────────────────────────── #
# Theme                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

GLASS_THEME = Theme(
    {
        "phase": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "skip": "dim",
        "info": "blue",
        "file": "dim cyan",
        "system": "bold magenta",
        "sample": "white",
        "metric": "yellow",
        "banner": "bold cyan",
    }
)

# ──────────────────────────────────────────────────────────────────────────── #
# ASCII Banner                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

BANNER = r"""
   ██████╗ ██╗      █████╗ ███████╗███████╗
  ██╔════╝ ██║     ██╔══██╗██╔════╝██╔════╝
  ██║  ███╗██║     ███████║███████╗███████╗
  ██║   ██║██║     ██╔══██║╚════██║╚════██║
  ╚██████╔╝███████╗██║  ██║███████║███████║
   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
"""

TAGLINE = "Global Long-context Agent Scoring System  v0.1.0"


# ──────────────────────────────────────────────────────────────────────────── #
# GlassConsole                                                                #
# ──────────────────────────────────────────────────────────────────────────── #


class GlassConsole:
    """Rich-based TUI wrapper with dual logging (console + file)."""

    def __init__(self):
        self.console = Console(theme=GLASS_THEME, highlight=False)
        self._file_handler: Optional[logging.FileHandler] = None
        self._start_time: Optional[float] = None

        # Configure root logger to use rich
        self._setup_console_logging()

    # ── Logging ─────────────────────────────────────────────────────────── #

    def _setup_console_logging(self):
        """Route Python logging through rich console."""
        root = logging.getLogger()
        # Remove existing handlers so we don't double-print
        for h in root.handlers[:]:
            root.removeHandler(h)

        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        rich_handler.setLevel(logging.DEBUG)
        root.addHandler(rich_handler)
        root.setLevel(logging.INFO)

    def setup_file_logging(self, run_dir: Path, level: str = "DEBUG"):
        """Add a file handler that logs everything to {run_dir}/glass.log."""
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "glass.log"

        self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._file_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_handler.setFormatter(fmt)
        logging.getLogger().addHandler(self._file_handler)

        return log_path

    # ── Banner ──────────────────────────────────────────────────────────── #

    def show_banner(self, experiment: str, run_id: str, dataset: str, systems: list[str]):
        """Display the GLASS ASCII art banner with run metadata."""
        from rich.console import Group

        banner_art = Text(BANNER, style="banner")

        meta = (
            f"  [bold]{TAGLINE}[/bold]\n"
            f"\n"
            f"  Experiment  [bold white]{experiment}[/bold white]\n"
            f"  Run ID      [bold white]{run_id}[/bold white]\n"
            f"  Dataset     [bold white]{dataset}[/bold white]\n"
            f"  Systems     [bold white]{', '.join(systems)}[/bold white]\n"
        )

        self.console.print(
            Panel(
                Group(banner_art, meta),
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self.console.print()
        self._start_time = time.time()

    # ── Phase Headers ───────────────────────────────────────────────────── #

    def phase_header(self, phase_num: int, title: str, detail: str = ""):
        """Print a prominent phase header."""
        label = f"◆  PHASE {phase_num} — {title}"
        if detail:
            label += f"    [dim]{detail}[/dim]"
        self.console.print()
        self.console.print(Rule(style="cyan"))
        self.console.print(f"  [phase]{label}[/phase]")
        self.console.print(Rule(style="cyan"))
        logging.getLogger("glass.pipeline").info("Phase %d: %s  %s", phase_num, title, detail)

    # ── Task Status ─────────────────────────────────────────────────────── #

    def task_success(self, system: str, sample_id: str, detail: str = ""):
        """Show a successful task completion."""
        extra = f"  [dim]{detail}[/dim]" if detail else ""
        self.console.print(f"  [success]✓[/success]  [system]{system:<22}[/system] [sample]sample_{sample_id}[/sample]{extra}")

    def task_skip(self, system: str, sample_id: str, reason: str = "checkpoint"):
        """Show a skipped task."""
        self.console.print(f"  [skip]↻  {system:<22} sample_{sample_id}  ({reason})[/skip]")

    def task_warning(self, system: str, sample_id: str, detail: str = ""):
        """Show a warning for a task."""
        extra = f"  [dim]{detail}[/dim]" if detail else ""
        self.console.print(f"  [warning]⚠[/warning]  [system]{system:<22}[/system] [sample]sample_{sample_id}[/sample]{extra}")
        logging.getLogger("glass.pipeline").warning(
            "%s / sample_%s: %s", system, sample_id, detail
        )

    def task_error(self, system: str, sample_id: str, detail: str = ""):
        """Show an error for a task."""
        extra = f"  [dim]{detail}[/dim]" if detail else ""
        self.console.print(f"  [error]✗[/error]  [system]{system:<22}[/system] [sample]sample_{sample_id}[/sample]{extra}")
        logging.getLogger("glass.pipeline").error(
            "%s / sample_%s: %s", system, sample_id, detail
        )

    # ── Errors & Warnings ───────────────────────────────────────────────── #

    def error_panel(self, title: str, body: str):
        """Show a prominent error panel for critical failures."""
        self.console.print()
        self.console.print(
            Panel(
                body,
                title=f"[error]✗ {title}[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        logging.getLogger("glass.pipeline").error("%s: %s", title, body)

    def warn_panel(self, title: str, body: str):
        """Show a prominent warning panel."""
        self.console.print(
            Panel(
                body,
                title=f"[warning]⚠ {title}[/warning]",
                border_style="yellow",
                padding=(0, 2),
            )
        )
        logging.getLogger("glass.pipeline").warning("%s: %s", title, body)

    # ── File Saves ──────────────────────────────────────────────────────── #

    def file_saved(self, label: str, path):
        """Show a dim file-save confirmation."""
        self.console.print(f"  [success]✓[/success]  Saved {label} → [file]{path}[/file]")

    # ── Progress Bar ────────────────────────────────────────────────────── #

    def make_progress(self, description: str = ""):
        """Create a rich Progress context manager for iteration tracking."""
        return Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    # ── Completion ──────────────────────────────────────────────────────── #

    def show_completion(self, run_dir: Path, stats: dict | None = None):
        """Show the final completion banner with timing and optional stats."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        mins, secs = divmod(int(elapsed), 60)

        self.console.print()
        self.console.print(Rule(style="green"))

        # Stats table if available
        if stats:
            table = Table(show_header=True, header_style="bold green", border_style="green", padding=(0, 2))
            table.add_column("Metric", style="bold")
            for sys_name in stats.get("systems", []):
                table.add_column(sys_name, justify="center")

            for metric_name, values in stats.get("metrics", {}).items():
                row = [metric_name]
                for sys_name in stats.get("systems", []):
                    val = values.get(sys_name)
                    if val is None:
                        row.append("[dim]—[/dim]")
                    elif isinstance(val, float):
                        row.append(f"{val:.3f}")
                    else:
                        row.append(str(val))
                table.add_row(*row)

            self.console.print(table)
            self.console.print()

        self.console.print(
            f"  [success]✓  Run complete[/success]   "
            f"Artifacts → [file]{run_dir}[/file]   "
            f"Duration: [bold]{mins}m {secs:02d}s[/bold]"
        )
        self.console.print(Rule(style="green"))
        self.console.print()

        logging.getLogger("glass.pipeline").info(
            "Run complete. Duration: %dm %02ds. Artifacts: %s", mins, secs, run_dir
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Singleton                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

_console: Optional[GlassConsole] = None


def get_console() -> GlassConsole:
    """Get or create the global GlassConsole instance."""
    global _console
    if _console is None:
        _console = GlassConsole()
    return _console
