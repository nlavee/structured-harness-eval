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

def select_runs(available_runs: list[str]) -> list[str]:
    """Interactive prompt for the user to select runs."""
    console.print(Panel("Available GLASS Runs", style="bold blue"))
    for i, run in enumerate(available_runs):
        console.print(f"[bold cyan][{i}][/bold cyan] {run}")
        
    console.print("\nEnter the indices of the runs you want to compare, separated by spaces (e.g., '0 2'):")
    while True:
        try:
            selection = input("> ").strip().split()
            indices = [int(x) for x in selection]
            selected_runs = [available_runs[i] for i in indices]
            if len(selected_runs) < 2:
                console.print("[bold red]Error: Please select at least 2 runs for comparison.[/bold red]")
                continue
            return selected_runs
        except (ValueError, IndexError):
            console.print("[bold red]Invalid input. Please enter valid space-separated indices.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting.[/bold yellow]")
            sys.exit(0)

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
        # Validate provided runs
        invalid_runs = [r for r in runs_to_compare if r not in available_runs]
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

    console.print(Panel(f"Initiating Research Harness\n[bold green]Output Directory:[/bold green] {out_dir}\n[bold blue]Runs:[/bold blue] {', '.join(runs_to_compare)}", title="GLASS Research Harness", style="bold magenta"))
    
    harness_dir = Path(__file__).parent
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
    
        # 1. Run Aggregator
        task1 = progress.add_task("[cyan]Step 1/3: Data Aggregation & AP-RH1 Enforcement...", total=None)
        agg_cmd = [sys.executable, str(harness_dir / "compare_runs.py"), "--runs"] + runs_to_compare + ["--runs-dir", str(runs_dir), "--out", str(out_dir / "aggregated_data.json")]
        subprocess.run(agg_cmd, check=True, capture_output=True)
        progress.update(task1, completed=100, description="[bold green]Step 1/3 Complete: Data Aggregation")
        
        # 2. Run Visualizer
        task2 = progress.add_task("[cyan]Step 2/3: Scientific Visualization & AP-RH2 Enforcement...", total=None)
        vis_cmd = [sys.executable, str(harness_dir / "visualizer.py"), "--data", str(out_dir / "aggregated_data.json"), "--out-dir", str(out_dir / "figures")]
        subprocess.run(vis_cmd, check=True, capture_output=True)
        progress.update(task2, completed=100, description="[bold green]Step 2/3 Complete: Scientific Visualization")
        
        # 3. Run Synthesizer
        if not args.skip_synthesis:
            task3 = progress.add_task(f"[cyan]Step 3/3: LLM Synthesis ({args.provider}/{args.model})...", total=None)
            synth_cmd = ["python", str(harness_dir / "synthesizer.py"), "--data", str(out_dir / "aggregated_data.json"), "--provider", args.provider, "--model", args.model, "--out-dir", str(out_dir)]
            subprocess.run(synth_cmd, check=True, capture_output=True)
            progress.update(task3, completed=100, description="[bold green]Step 3/3 Complete: LLM Synthesis")
        else:
            console.print("[yellow]Step 3: LLM Qualitative Synthesis [SKIPPED][/yellow]")
            
        # 4. Run Vision Interpreter
        if not args.skip_synthesis:
            task4 = progress.add_task(f"[cyan]Step 4/4: Vision Interpretation ({args.provider}/{args.model})...", total=None)
            vision_cmd = ["python", str(harness_dir / "vision_interpreter.py"), "--figures-dir", str(out_dir / "figures"), "--provider", args.provider, "--model", args.model, "--out-dir", str(out_dir)]
            subprocess.run(vision_cmd, check=True, capture_output=True)
            progress.update(task4, completed=100, description="[bold green]Step 4/4 Complete: Vision Interpretation")
        else:
            console.print("[yellow]Step 4: Vision Interpretation [SKIPPED][/yellow]")
            
    console.print(f"\n[bold green]✨ Research Harness pipeline complete.[/bold green] Check [bold cyan]{out_dir}/[/bold cyan] for artifacts.")

if __name__ == "__main__":
    main()
