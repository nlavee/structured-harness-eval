import os

from dotenv import load_dotenv

# Load .env file if present (does not override existing env vars)
load_dotenv(override=False)

import click
import yaml

# Ensure modules are loaded to populate registry
import glass.datasets.aalcr  # noqa: F401
import glass.metrics.error_rate  # noqa: F401
import glass.metrics.exact_match  # noqa: F401
import glass.metrics.latency  # noqa: F401
import glass.metrics.refusal  # noqa: F401
import glass.metrics.soft_recall  # noqa: F401
import glass.metrics.verbosity  # noqa: F401
import glass.systems.claude  # noqa: F401
import glass.systems.codex  # noqa: F401
import glass.systems.gemini  # noqa: F401
import glass.systems.structured_harness  # noqa: F401
import glass.systems.stub  # noqa: F401

from glass.config.schema import Config
from glass.pipeline import Pipeline


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--resume", type=click.STRING, help="Run ID to resume")
def run(config_path, resume):
    """Run the evaluation pipeline."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = Config(**config_data)

    run_id = None
    if resume:
        # If full path provided, extract ID
        run_id = os.path.basename(resume.rstrip("/"))

    pipeline = Pipeline(config, run_id=run_id)
    pipeline.run()


if __name__ == "__main__":
    cli()
