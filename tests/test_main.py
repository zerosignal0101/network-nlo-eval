"""Test cases for the __main__ module."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from network_nlo_eval import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.main, "--help")
    assert result.exit_code == 0


def _write_topology(path: Path) -> Path:
    """写一个三节点线形拓扑，供 CLI 冒烟测试使用。"""
    payload = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
        "edges": [
            {"source": 1, "target": 2, "weight": 80.0},
            {"source": 2, "target": 3, "weight": 80.0},
        ],
    }
    topology_file = path / "topology.json"
    topology_file.write_text(json.dumps(payload), encoding="utf-8")
    return topology_file


def test_simulate_rejects_slot_width_wider_than_channel(runner: CliRunner, tmp_path: Path) -> None:
    """槽宽不得超过信道间隔。"""
    topology_file = _write_topology(tmp_path)
    result = runner.invoke(
        __main__.main,
        ["simulate", "-t", str(topology_file), "--channel-spacing-ghz", "50", "--slot-width-ghz", "60"],
    )
    assert result.exit_code != 0
    assert "no wider than the channel spacing" in result.output


def test_simulate_rejects_non_integer_slot_ratio(runner: CliRunner, tmp_path: Path) -> None:
    """信道间隔必须是槽宽的整数倍。"""
    topology_file = _write_topology(tmp_path)
    result = runner.invoke(
        __main__.main,
        ["simulate", "-t", str(topology_file), "--channel-spacing-ghz", "50", "--slot-width-ghz", "30"],
    )
    assert result.exit_code != 0
    assert "integer multiple" in result.output


def test_simulate_runs_with_flexible_grid(runner: CliRunner, tmp_path: Path) -> None:
    """带 --slot-width-ghz 时走灵活栅格分配器并正常落盘。"""
    topology_file = _write_topology(tmp_path)
    output_dir = tmp_path / "out"
    result = runner.invoke(
        __main__.main,
        [
            "simulate",
            "-t",
            str(topology_file),
            "-s",
            "5",
            "--num-channels",
            "8",
            "--slot-width-ghz",
            "12.5",
            "-o",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Flexible grid enabled" in result.output

    report = json.loads((output_dir / "simulation_results.json").read_text(encoding="utf-8"))
    assert "simulation_metrics" in report
