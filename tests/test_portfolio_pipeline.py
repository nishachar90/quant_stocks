from pathlib import Path

import pytest

import scripts.run_portfolio_pipeline as pipeline


def test_pipeline_steps_are_defined():
    """Verify that all required pipeline stages exist."""

    expected_steps = [
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

    assert pipeline.PIPELINE_STEPS == expected_steps


def test_pipeline_scripts_exist():
    """Verify that every pipeline script exists."""

    for _, script_path in pipeline.PIPELINE_STEPS:
        full_path = pipeline.PROJECT_ROOT / script_path

        assert full_path.exists(), (
            f"Missing pipeline script: {full_path}"
        )


def test_run_step_success(monkeypatch):
    """Verify that a successful subprocess completes."""

    calls = []

    class MockResult:
        returncode = 0

    def mock_run(
        command,
        cwd,
        env,
        check,
    ):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "check": check,
            }
        )

        return MockResult()

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        mock_run,
    )

    pipeline.run_step(
        step_name="Test Step",
        script_path="scripts/generate_portfolio.py",
    )

    assert len(calls) == 1

    assert calls[0]["command"][0] == (
        pipeline.sys.executable
    )

    assert (
        calls[0]["command"][1]
        == str(
            pipeline.PROJECT_ROOT
            / "scripts"
            / "generate_portfolio.py"
        )
    )

    assert (
        calls[0]["cwd"]
        == pipeline.PROJECT_ROOT
    )

    assert calls[0]["check"] is False

    assert (
        calls[0]["env"]["PYTHONPATH"]
        .split(":")[0]
        == str(pipeline.PROJECT_ROOT)
    )


def test_run_step_missing_script():
    """Verify that a missing script raises an error."""

    with pytest.raises(
        FileNotFoundError,
        match="Pipeline script not found",
    ):
        pipeline.run_step(
            step_name="Missing Step",
            script_path="scripts/does_not_exist.py",
        )


def test_run_step_failure(monkeypatch):
    """Verify that a failed pipeline stage stops execution."""

    class MockResult:
        returncode = 1

    def mock_run(
        command,
        cwd,
        env,
        check,
    ):
        return MockResult()

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        mock_run,
    )

    with pytest.raises(
        RuntimeError,
        match="Pipeline step failed",
    ):
        pipeline.run_step(
            step_name="Failing Step",
            script_path="scripts/generate_portfolio.py",
        )


def test_run_pipeline_executes_all_steps(
    monkeypatch,
):
    """Verify that all pipeline stages execute in order."""

    executed_steps = []

    def mock_run_step(
        step_name,
        script_path,
    ):
        executed_steps.append(
            (
                step_name,
                script_path,
            )
        )

    monkeypatch.setattr(
        pipeline,
        "run_step",
        mock_run_step,
    )

    pipeline.run_pipeline()

    assert (
        executed_steps
        == pipeline.PIPELINE_STEPS
    )
