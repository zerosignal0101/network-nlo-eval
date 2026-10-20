"""Command-line interface."""

import json
from pathlib import Path

import click
import networkx as nx
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model, describe_models
from network_nlo_eval.models.numba_kernels import _dbm_to_w
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.comparison import TopologyQoTComparator
from network_nlo_eval.rwa.allocators import KSPFirstFitAllocator, KSPFirstFitFlexAllocator
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import BaseRWAAllocator, QoTValidator

# 导入核心模拟器
from network_nlo_eval.simulation.engine import SimulatorEngine
from network_nlo_eval.simulation.reporter import MetricsCollector
from network_nlo_eval.simulation.traffic import generate_services


@click.group()
@click.version_option()
def main() -> None:
    """Network NLO Eval."""
    pass


@main.command(name="models")
def list_models() -> None:
    """List the registered QoT models as JSON."""
    click.echo(json.dumps(describe_models(), ensure_ascii=False, indent=2))


def _load_topology(topology_file: Path) -> NetworkTopology:
    """加载 JSON 图并应用可由节点/链路属性覆盖的物理默认值."""
    with open(topology_file, encoding="utf-8") as file:
        network_obj = json.load(file)
    network_raw = nx.node_link_graph(network_obj, edges="edges")
    return NetworkTopology(
        network_raw=network_raw,
        default_fiber_config=FiberSpanConfig(length_km=100.0),
        default_edfa_config=EDFAConfig(),
        default_roadm_config=ROADMConfig(),
    )


@main.command()
@click.option(
    "-t",
    "--topology-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="assets/example_pan_europe_network.json",
    help="Path to the NetworkX JSON topology file.",
)
@click.option(
    "-s",
    "--service-num",
    type=int,
    default=1000,
    help="Number of services to generate.",
)
@click.option(
    "--avg-arrival-interval",
    type=float,
    default=10.0,
    help="Average arrival interval for services (time units).",
)
@click.option(
    "--avg-holding-time",
    type=float,
    default=400.0,
    help="Average holding time for services (time units).",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="results",
    help="Directory to save simulation results.",
)
@click.option("--num-channels", type=int, default=80, help="Number of WDM channels.")
@click.option("--channel-spacing-ghz", type=float, default=50.0, help="Channel spacing in GHz.")
@click.option("--center-freq-thz", type=float, default=193.1, help="Center frequency in THz.")
@click.option(
    "--max-ksp-paths",
    type=int,
    default=5,
    help="Maximum number of KSP paths to precompute.",
)
@click.option(
    "--slot-width-ghz",
    type=float,
    default=None,
    help="Enable flexible-grid slot allocation with this slot width (e.g. 12.5).",
)
@click.option(
    "--qot-model",
    "qot_model",
    default="ecz",
    show_default=True,
    help="Registered QoT model used for SNR checks (see 'network-nlo-eval models').",
)
def simulate(
    topology_file: Path,
    service_num: int,
    avg_arrival_interval: float,
    avg_holding_time: float,
    output_dir: Path,
    num_channels: int,
    channel_spacing_ghz: float,
    center_freq_thz: float,
    max_ksp_paths: int,
    slot_width_ghz: float | None,
    qot_model: str,
) -> None:
    """Run a dynamic RWA simulation."""
    click.echo(f"Starting simulation with topology: {topology_file}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = output_dir / "simulation_results.json"

    # 1. 加载拓扑
    try:
        network_topology = _load_topology(topology_file)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        click.secho(f"Error loading topology file: {exc}", fg="red")
        return

    # 2. 初始化核心组件
    # 频谱网格
    spectrum_grid = SpectrumGrid(
        num_channels=num_channels,
        channel_spacing_hz=channel_spacing_ghz * 1e9,
        center_frequency_hz=center_freq_thz * 1e12,
    )

    # 物理层快速评估模型：由注册表选择，默认 ECZ。
    try:
        gn_evaluator = create_model(
            qot_model,
            spectrum_grid,
            network_topology.get_default_fiber_config(),
        )
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    # QoT 验证器
    qot_validator = QoTValidator(
        gn_evaluator=gn_evaluator,
        network_topology=network_topology,
        spectrum_grid=spectrum_grid,
    )

    # 路径计算缓存
    path_cache = PathCache(network_topology, max_ksp_paths)

    # 灵活栅格（可选）：槽宽须能整除信道间隔，折算率由两者之比决定。
    slot_width_hz: float | None = None
    slots_per_channel = 1
    if slot_width_ghz is not None:
        if slot_width_ghz <= 0 or slot_width_ghz > channel_spacing_ghz:
            raise click.ClickException("--slot-width-ghz must be positive and no wider than the channel spacing.")
        ratio = channel_spacing_ghz / slot_width_ghz
        if not np.isclose(ratio, round(ratio)):
            raise click.ClickException("--channel-spacing-ghz must be an integer multiple of --slot-width-ghz.")
        slots_per_channel = int(round(ratio))
        slot_width_hz = slot_width_ghz * 1e9

    # RWA 分配器
    if slot_width_hz is None:
        allocator: BaseRWAAllocator = KSPFirstFitAllocator(
            path_cache=path_cache,
            qot_validator=qot_validator,
            spectrum_grid=spectrum_grid,
        )
    else:
        allocator = KSPFirstFitFlexAllocator(
            path_cache=path_cache,
            qot_validator=qot_validator,
            spectrum_grid=spectrum_grid,
            network_topology=network_topology,
            slots_per_channel=slots_per_channel,
        )
        click.echo(
            f"Flexible grid enabled: {slot_width_ghz} GHz slots, "
            f"{slots_per_channel} slots per {channel_spacing_ghz} GHz channel."
        )

    # 网络动态状态
    network_state = NetworkState(
        network_topology=network_topology,
        spectrum_grid=spectrum_grid,
        slots_per_channel=slots_per_channel,
    )

    # 流量生成器
    services = generate_services(
        path_cache=path_cache,
        service_num=service_num,
        avg_arrival_interval=avg_arrival_interval,
        avg_holding_time=avg_holding_time,
        slot_width_hz=slot_width_hz,
    )
    click.echo(f"Generated {len(services)} services.")

    # 性能指标收集器
    metrics_collector = MetricsCollector(
        network_topology=network_topology,
        spectrum_grid=spectrum_grid,
        total_incoming_services=len(services),
    )

    # 3. 运行仿真
    simulator = SimulatorEngine(
        allocator=allocator,
        network_state=network_state,
        metrics_collector=metrics_collector,
        incoming_services=services,
    )
    click.echo("Running simulation...")
    simulator.run()
    click.echo("Simulation finished.")

    # 4. 生成报告并导出
    report_data = metrics_collector.generate_report(network_state)
    report_data["topology"] = network_topology.export_to_dict()

    with open(results_filepath, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    click.echo(f"Simulation results saved to: {results_filepath}")
    click.echo("\n--- Simulation Summary ---")
    click.echo(f"Blocking Rate: {report_data['simulation_metrics']['blocking_rate']:.4f}")
    click.echo(f"Wavelength Utilization (End State): {report_data['simulation_metrics']['wavelength_utilization']:.4f}")
    click.echo(
        f"Average Hop Count for Allocated Services: {report_data['simulation_metrics']['average_hop_count']:.2f}"
    )
    click.echo(
        f"Average Wavelength Fragmentation Index (End State): "
        f"{report_data['simulation_metrics']['average_wavelength_fragmentation_index']:.4f}"
    )
    click.echo("--------------------------")


@main.command()
@click.option("-t", "--topology-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--path", "path_text", required=True, help="Comma-separated original topology node IDs.")
@click.option("--channel", "channels", type=int, multiple=True, help="Active zero-based channel; repeat as needed.")
@click.option("--launch-power-dbm", type=float, default=0.0, show_default=True)
@click.option("--num-channels", type=int, default=80, show_default=True)
@click.option("--channel-spacing-ghz", type=float, default=50.0, show_default=True)
@click.option("--center-freq-thz", type=float, default=193.1, show_default=True)
@click.option(
    "--qot-model",
    "qot_model",
    default="ecz",
    show_default=True,
    help="Registered QoT model for the statistical side (see 'network-nlo-eval models').",
)
@click.option("--waveform/--statistical-only", default=False, help="Also run the numerical waveform engine.")
@click.option("--num-symbols", type=int, default=1024, show_default=True)
@click.option("-o", "--output-file", type=click.Path(dir_okay=False, path_type=Path))
def compare(
    topology_file: Path,
    path_text: str,
    channels: tuple[int, ...],
    launch_power_dbm: float,
    num_channels: int,
    channel_spacing_ghz: float,
    center_freq_thz: float,
    qot_model: str,
    waveform: bool,
    num_symbols: int,
    output_file: Path | None,
) -> None:
    """Compare statistical and optional waveform QoT on one topology path."""
    try:
        topology = _load_topology(topology_file)
        original_path = [int(value.strip()) for value in path_text.split(",") if value.strip()]
        path = [topology.get_internal_node_id(node_id) for node_id in original_path]
        grid = SpectrumGrid(num_channels, channel_spacing_ghz * 1e9, center_freq_thz * 1e12)
        active_channels = channels or (num_channels // 2,)
        if any(index < 0 or index >= num_channels for index in active_channels):
            raise ValueError("Channel index is outside the configured spectrum grid.")
        launch_profile = np.zeros(num_channels, dtype=np.float64)
        launch_profile[list(active_channels)] = _dbm_to_w(launch_power_dbm)
        gn_evaluator = create_model(qot_model, grid, topology.get_default_fiber_config())
        result = TopologyQoTComparator(topology, grid, gn_evaluator=gn_evaluator).evaluate(
            path, launch_profile, run_waveform=waveform, num_symbols=num_symbols
        )
        payload = result.to_dict()
        payload["original_path"] = original_path
        output = json.dumps(payload, ensure_ascii=False, indent=2)
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output + "\n", encoding="utf-8")
            click.echo(f"Comparison saved to: {output_file}")
        else:
            click.echo(output)
    except (ImportError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--case", "case_name", required=True, help="Benchmark case (see benchmarks.cases.BENCH_CASES).")
@click.option("--model", "models", multiple=True, required=True, help="Registered QoT model; repeat for several.")
@click.option("--reference", default=None, help="Reference model for accuracy deltas, e.g. 'numerical'.")
@click.option("--power-step-db", type=float, default=1.0, show_default=True, help="Launch-power sweep step.")
@click.option("--reps", type=int, default=25, show_default=True, help="Warm timing repetitions.")
@click.option("-o", "--output-file", type=click.Path(dir_okay=False, path_type=Path))
def bench(
    case_name: str,
    models: tuple[str, ...],
    reference: str | None,
    power_step_db: float,
    reps: int,
    output_file: Path | None,
) -> None:
    """Run a benchmark case across models and report accuracy plus speed."""
    from network_nlo_eval.benchmarks.cases import build_case
    from network_nlo_eval.benchmarks.runner import accuracy_vs_reference, run_case

    try:
        case = build_case(case_name)
        powers = tuple(np.arange(-4.0, 6.0 + 1e-9, power_step_db))
        runs = {}
        for name in models:
            run = run_case(name, case.name, powers_dbm=powers, reps=reps)
            runs[name] = run
        payload = {
            "case": {
                "name": case.name,
                "bands": case.bands,
                "num_channels": case.num_channels,
                "channel_spacing_ghz": case.channel_spacing_hz / 1e9,
                "symbol_rate_ghz": case.symbol_rate_hz / 1e9,
                "span_count": case.span_count,
                "span_length_km": case.span_length_km,
                "edfa_noise_figure_db": case.edfa_noise_figure_db,
                "total_power_dbm": case.total_power_dbm,
                "powers_dbm": list(powers),
            },
            "runs": {name: run.to_dict() for name, run in runs.items()},
            "accuracy": {},
        }
        if reference is not None:
            if reference not in runs:
                raise click.ClickException(f"--reference {reference!r} must also be passed via --model.")
            for name, run in runs.items():
                if name == reference:
                    continue
                payload["accuracy"][name] = accuracy_vs_reference(run, runs[reference])
        echo_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(echo_payload + "\n", encoding="utf-8")
            click.echo(f"Benchmark saved to: {output_file}")
        else:
            click.echo(echo_payload)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main(prog_name="network-nlo-eval")
