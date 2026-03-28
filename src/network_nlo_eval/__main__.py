"""Command-line interface."""

import json
from pathlib import Path

import click
import networkx as nx

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.allocators import KSPFirstFitAllocator
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import QoTValidator

# 导入核心模拟器
from network_nlo_eval.simulation.engine import SimulatorEngine
from network_nlo_eval.simulation.reporter import MetricsCollector
from network_nlo_eval.simulation.traffic import generate_services


@click.group()
@click.version_option()
def main() -> None:
    """Network NLO Eval."""
    pass


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
) -> None:
    """Run a dynamic RWA simulation."""
    click.echo(f"Starting simulation with topology: {topology_file}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = output_dir / "simulation_results.json"

    # 1. 加载拓扑
    try:
        with open(topology_file, encoding="utf-8") as f:
            network_obj = json.load(f)
        network_raw = nx.node_link_graph(network_obj, edges="edges")
    except Exception as e:
        click.secho(f"Error loading topology file: {e}", fg="red")
        return

    # 2. 初始化核心组件
    # 频谱网格
    spectrum_grid = SpectrumGrid(
        num_channels=num_channels,
        channel_spacing_hz=channel_spacing_ghz * 1e9,
        center_frequency_hz=center_freq_thz * 1e12,
    )

    # 网络拓扑管理 (包含光纤/EDFA/ROADM 配置)
    # 假设所有光纤链路使用相同的默认配置，这里可以根据网络拓扑中的link data进行更细致的配置
    # 也可以在 NetworkTopology 类中直接从 graph attributes 读取
    default_fiber_config = FiberSpanConfig(
        length_km=100.0,  # This will be overwritten by actual link lengths
        attenuation_db_km_ref=0.2,  # dB/km at reference wavelength
        dispersion_parameter_d=17.0,  # ps/(nm*km) at reference wavelength
        dispersion_slope_s=0.06,  # ps/(nm^2*km) at reference wavelength
        nonlinear_index_n2=2.6e-20,  # m^2/W
        effective_area_um2_ref=80.0,  # um^2 at reference wavelength
        aeff_slope_um2_nm=0.05,  # um^2 per nm
        reference_wavelength_nm=1550.0,  # nm
    )
    # 假设所有节点都配备默认的 EDFA 和 ROADM
    default_edfa_config = EDFAConfig(target_gain_db=20.0, noise_figure_db=5.0)
    default_roadm_config = ROADMConfig(insertion_loss_db=12.0, filtering_penalty_db=0.5)

    network_topology = NetworkTopology(
        network_raw=network_raw,
        default_fiber_config=default_fiber_config,
        default_edfa_config=default_edfa_config,
        default_roadm_config=default_roadm_config,
    )

    # 物理层快速评估模型
    gn_evaluator = MultiBandISRSGN(
        grid=spectrum_grid,
        ref_fiber_config=network_topology.get_default_fiber_config(),  # Use the default fiber config as reference
    )

    # QoT 验证器
    qot_validator = QoTValidator(
        gn_evaluator=gn_evaluator,
        network_topology=network_topology,
        spectrum_grid=spectrum_grid,
    )

    # 路径计算缓存
    path_cache = PathCache(network_topology, max_ksp_paths)

    # RWA 分配器
    allocator = KSPFirstFitAllocator(
        path_cache=path_cache,
        qot_validator=qot_validator,
        spectrum_grid=spectrum_grid,
    )

    # 网络动态状态
    network_state = NetworkState(
        network_topology=network_topology,
        spectrum_grid=spectrum_grid,
    )

    # 流量生成器
    services = generate_services(
        path_cache=path_cache,
        service_num=service_num,
        avg_arrival_interval=avg_arrival_interval,
        avg_holding_time=avg_holding_time,
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


if __name__ == "__main__":
    main(prog_name="network-nlo-eval")
