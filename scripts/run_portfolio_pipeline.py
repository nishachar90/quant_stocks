from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_STEPS = [
    (
        "Portfolio Construction",
        "scripts/generate_portfolio.py",
    ),
    (
        "Portfolio Backtest",
        "scripts/generate_portfolio_backtest.py",
    ),
    (
        "Benchmark Analysis",
        "scripts/generate_benchmark_analysis.py",
    ),
    (
        "Portfolio Evaluation & Attribution",
        "scripts/generate_portfolio_evaluation.py",
    ),
]


def run_step(
    step_name: str,
    script_path: str,
) -> None:
    """Run one pipeline stage."""

    print()
    print("=" * 70)
    print(step_name.upper())
    print("=" * 70)
    print(f"Running: {script_path}")

    script = PROJECT_ROOT / script_path

    if not script.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {script}"
        )

    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    if existing_pythonpath:
        environment["PYTHONPATH"] = (
            f"{PROJECT_ROOT}{os.pathsep}"
            f"{existing_pythonpath}"
        )
    else:
        environment["PYTHONPATH"] = str(
            PROJECT_ROOT
        )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: "
            f"{step_name} "
            f"(exit code {result.returncode})"
        )

    print()
    print(f"{step_name}: SUCCESS")


def run_pipeline() -> None:
    """Run the complete portfolio research pipeline."""

    print("=" * 70)
    print("NIFTY 50 END-TO-END PORTFOLIO PIPELINE")
    print("=" * 70)

    print()
    print("Pipeline:")
    print("1. Portfolio Construction")
    print("2. Portfolio Backtest")
    print("3. Benchmark Analysis")
    print("4. Portfolio Evaluation & Attribution")

    for step_name, script_path in PIPELINE_STEPS:
        run_step(
            step_name=step_name,
            script_path=script_path,
        )

    print()
    print("=" * 70)
    print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
