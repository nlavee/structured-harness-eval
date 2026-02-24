import argparse
import sys
import subprocess
from pathlib import Path
import logging

from datetime import datetime
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logging.basicConfig(level=logging.ERROR) # Let Rich handle the UI
logger = logging.getLogger("ResearchHarness")

def get_available_runs(runs_dir: Path) -> list[str]:
    """Returns a list of valid run IDs."""
    if not runs_dir.exists():
        return []
    
    runs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and (d / "statistics.json").exists() and (d / "results.csv").exists():
            runs.append(d.name)
    return sorted(runs, reverse=True)

def parse_run_arg(arg: str) -> tuple[str, str | None]:
    """Parses a run argument in run_id:alias format."""
    if ":" in arg:
        parts = arg.split(":", 1)
        return parts[0], parts[1]
    return arg, None

def select_runs(available_runs: list[str]) -> list[str]:
    """Interactive prompt for the user to select runs."""
    console.print(Panel("Available GLASS Runs", style="bold blue"))
    for i, run in enumerate(available_runs):
        console.print(f"[bold cyan][{i}][/bold cyan] {run}")
        
    console.print("\nEnter the indices of the runs you want to compare, separated by spaces (e.g., '0 2'):")
    console.print("You can also add an alias after the index with a colon (e.g., '0:G3Pro 2:G3Flash').")
    while True:
        try:
            selection = input("> ").strip().split()
            selected_runs = []
            for item in selection:
                if ":" in item:
                    idx_str, alias = item.split(":", 1)
                    idx = int(idx_str)
                    selected_runs.append(f"{available_runs[idx]}:{alias}")
                else:
                    idx = int(item)
                    selected_runs.append(available_runs[idx])

            if len(selected_runs) < 2:
                console.print("[bold red]Error: Please select at least 2 runs for comparison.[/bold red]")
                continue
            return selected_runs
        except (ValueError, IndexError):
            console.print("[bold red]Invalid input. Please enter valid space-separated indices (and optional aliases).[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting.[/bold yellow]")
            sys.exit(0)

def run_step(command: list[str], description: str, progress, task_id):
    """Executes a subprocess command, streaming stdout to the console."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout
        text=True,
        bufsize=1, # Line buffered
        universal_newlines=True
    )

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            clean_line = line.strip()
            # Heuristic to colorize or format specific log lines if needed
            if "INFO" in clean_line:
                progress.console.print(f"  [dim]{clean_line}[/dim]")
            elif "WARNING" in clean_line:
                 progress.console.print(f"  [bold yellow]{clean_line}[/bold yellow]")
            elif "ERROR" in clean_line:
                 progress.console.print(f"  [bold red]{clean_line}[/bold red]")
            else:
                 progress.console.print(f"  {clean_line}")

    if process.returncode != 0:
        progress.console.print(f"[bold red]Step failed with exit code {process.returncode}[/bold red]")
        sys.exit(process.returncode)
    
    progress.update(task_id, completed=100, description=f"[bold green]{description} Complete")

def main():
    parser = argparse.ArgumentParser(description="GLASS Research Harness Orchestrator.")
    parser.add_argument("--runs", nargs="*", help="Run IDs to compare. If empty, prompts interactively.")
    parser.add_argument("--runs-dir", default="runs", help="Base directory for runs.")
    parser.add_argument("--provider", default="anthropic", help="The LiteLLM provider for synthesis (e.g., anthropic, gemini, openai).")
    parser.add_argument("--model", default="claude-3-5-sonnet-20240620", help="The exact model string for synthesis.")
    parser.add_argument("--skip-synthesis", action="store_true", help="Skip the LLM synthesis step.")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    available_runs = get_available_runs(runs_dir)
    
    if not available_runs:
        logger.error(f"No valid runs found in {runs_dir}")
        sys.exit(1)
        
    runs_to_compare = args.runs
    if not runs_to_compare:
        runs_to_compare = select_runs(available_runs)
    else:
        # Validate provided runs (ignoring aliases for check)
        invalid_runs = []
        for r_arg in runs_to_compare:
            r_id, _ = parse_run_arg(r_arg)
            if r_id not in available_runs:
                invalid_runs.append(r_id)
        
        if invalid_runs:
            logger.error(f"Invalid runs provided: {invalid_runs}")
            sys.exit(1)
            
    if len(runs_to_compare) < 2:
        console.print("[bold red]Error: Must specify at least 2 runs to compare.[/bold red]")
        sys.exit(1)

    # 0. Generate Output Directory & Save Command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"research_insights/run_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd_str = " ".join(sys.argv)
    with open(out_dir / "command.txt", "w") as f:
        f.write(cmd_str + "\n")

    console.print(Panel(f"Initiating Research Harness\n[bold green]Output Directory:[/bold green] {out_dir}\n[bold blue]Runs:[/bold blue] {', '.join(runs_to_compare)}\n[bold cyan]Synthesis Model:[/bold cyan] {args.provider}/{args.model}", title="GLASS Research Harness", style="bold magenta"))
    
    harness_dir = Path(__file__).parent
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
    
        # 1. Run Aggregator
        task1 = progress.add_task("[cyan]Step 1/4: Data Aggregation & AP-RH1 Enforcement...", total=None)
        agg_cmd = [sys.executable, str(harness_dir / "compare_runs.py"), "--runs"] + runs_to_compare + ["--runs-dir", str(runs_dir), "--out", str(out_dir / "aggregated_data.json")]
        run_step(agg_cmd, "Step 1/4: Data Aggregation", progress, task1)
        
        # 2. Run Visualizer
        task2 = progress.add_task("[cyan]Step 2/4: Scientific Visualization & AP-RH2 Enforcement...", total=None)
        vis_cmd = [sys.executable, str(harness_dir / "visualizer.py"), "--data", str(out_dir / "aggregated_data.json"), "--out-dir", str(out_dir / "figures")]
        run_step(vis_cmd, "Step 2/4: Scientific Visualization", progress, task2)
        
        # 3. Run Synthesizer
        if not args.skip_synthesis:
            task3 = progress.add_task(f"[cyan]Step 3/4: LLM Synthesis ({args.provider}/{args.model})...", total=None)
            synth_cmd = ["python", str(harness_dir / "synthesizer.py"), "--data", str(out_dir / "aggregated_data.json"), "--provider", args.provider, "--model", args.model, "--out-dir", str(out_dir)]
            run_step(synth_cmd, "Step 3/4: LLM Synthesis", progress, task3)
        else:
            console.print("[yellow]Step 3: LLM Qualitative Synthesis [SKIPPED][/yellow]")
            
        # 4. Run Vision Interpreter
        if not args.skip_synthesis:
            task4 = progress.add_task(f"[cyan]Step 4/4: Vision Interpretation ({args.provider}/{args.model})...", total=None)
            vision_cmd = ["python", str(harness_dir / "vision_interpreter.py"), "--figures-dir", str(out_dir / "figures"), "--provider", args.provider, "--model", args.model, "--out-dir", str(out_dir)]
            run_step(vision_cmd, "Step 4/4: Vision Interpretation", progress, task4)
        else:
            console.print("[yellow]Step 4: Vision Interpretation [SKIPPED][/yellow]")
            
    console.print(f"\n[bold green]✨ Research Harness pipeline complete.[/bold green] Check [bold cyan]{out_dir}/[/bold cyan] for artifacts.")

if __name__ == "__main__":
    main()
