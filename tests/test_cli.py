"""Command-line model listing, comparison and simulation."""

import json

import networkx as nx
from click.testing import CliRunner

from network_nlo_eval.__main__ import main


def test_cli_models_compare_and_simulate(tmp_path) -> None:
    runner = CliRunner()
    models = runner.invoke(main, ["models"])
    assert models.exit_code == 0, models.output
    assert any(item["name"] == "pcfm_h3_raman" for item in json.loads(models.output))

    graph = nx.Graph()
    graph.add_edge(10, 20, weight=20)
    topology = tmp_path / "topology.json"
    topology.write_text(
        json.dumps(nx.node_link_data(graph, edges="edges")), encoding="utf-8"
    )
    comparison = runner.invoke(
        main,
        [
            "compare",
            "-t",
            str(topology),
            "--path",
            "10,20",
            "--num-channels",
            "2",
            "--channel",
            "0",
        ],
    )
    assert comparison.exit_code == 0, comparison.output
    assert json.loads(comparison.output)["original_path"] == [10, 20]

    output = tmp_path / "output"
    simulation = runner.invoke(
        main,
        [
            "simulate",
            "-t",
            str(topology),
            "-o",
            str(output),
            "--service-num",
            "2",
            "--num-channels",
            "4",
            "--slot-width-ghz",
            "12.5",
        ],
    )
    assert simulation.exit_code == 0, simulation.output
    report = json.loads(
        (output / "simulation_results.json").read_text(encoding="utf-8")
    )
    assert "simulation_metrics" in report
    assert len(report["service_events"]) >= 2

    model_output = tmp_path / "semrau-output"
    simulation = runner.invoke(
        main,
        [
            "simulate",
            "-t",
            str(topology),
            "-o",
            str(model_output),
            "--service-num",
            "2",
            "--num-channels",
            "4",
            "--qot-model",
            "semrau2019",
        ],
    )
    assert simulation.exit_code == 0, simulation.output
    model_report = json.loads(
        (model_output / "simulation_results.json").read_text(encoding="utf-8")
    )
    assert model_report["simulation_metrics"]["total_incoming_services"] == 2
    assert (
        model_report["simulation_metrics"]["successfully_allocated_service_count"] > 0
    )


def test_default_topology_is_packaged() -> None:
    runner = CliRunner()
    command = main.commands["simulate"]
    topology_option = next(p for p in command.params if p.name == "topology_file")
    assert topology_option.default.is_file()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "simulate" in help_result.output
