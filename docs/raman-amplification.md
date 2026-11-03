# 拉曼放大支持：集总等效与论文 hybrid 对照模型

日期：2026-11-03
分支：`feat/raman-amp`（基于 `multi-model-dev` d949f7b）

本文档区分两个层次，避免把近似当成论文模型：

1. **集总等效 DRA**（`models/raman.py`，注册名 `raman_lumped`）：泵浦未耗尽、
   跨末乘增益、跨内 NLI 不变。用于快速链路预算与泵浦初值筛选。
2. **论文 hybrid 对照模型**（`models/hybrid_raman.py`，注册名
   `hybrid_raman_paper`）：论文 Eq. (2)/(4) 的 coupled signal/pump/ASE ODE
   子集，NLI 仍是近似（不是 Eq. (6)-(9) 全式）。用于检验 lumped 近似到底
   偏了多少。

## 1. 现状

| 路径 | 拉曼泵浦支持 | 说明 |
|---|---|---|
| 波形物理 `physics_nlo_eval.RamanPump` | **有** | Gaussian 增益 + ASE，FW/BW，任意泵；本身也是**未耗尽集总**块，不是论文 coupled ODE |
| `FiberSpanConfig.raman_pumps` | **配置有** | 解析 lumped / paper 模型都可读取 |
| PCFM/H3 解析 | **原无** | 只做信道间 ISRS；`Raman pumping is not modelled` |
| `raman_lumped` | **有** | 跨末乘 on-off 增益，Raman ASE 在 EDFA 前注入 |
| `hybrid_raman_paper` | **有（子集）** | Eq. (2)/(4) ODE；NLI 用 PCFM 核 + ODE 剖面近似 |
| cGN | 部分 | `k_bar` 只是泵浦功率占比上限，不是谱分辨拉曼放大 |
| D'Amico2022 | 无 | 明确不做 DRA |

## 2. 集总等效组件

`src/network_nlo_eval/models/raman.py`

- `RamanGainModel`
  - Gaussian（默认，峰 13.2 THz，σ=2.12 THz）或 CZ 线性截断谱；
  - 任意 FW/BW 泵；
  - 未耗尽泵浦有效长度；
  - 逐信道 `A_eff` 与泵浦/信号光子能量因子；
  - Raman ASE 沿跨一维积分。
- `RamanAmplifier`
  - `gain/gain_db/ase_w/apply`；
  - 集总在跨段末端，随后由 EDFA 补偿剩余损耗。

PCFM opt-in：

```python
model = create_model(
    "pcfm", grid, fiber,
    raman_model="lumped",          # 或 raman_pumps=[...] 自动启用
    raman_pumps=[RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )],
    raman_gain_model="gaussian",   # 或 "linear"
    raman_ase=True,
)
```

默认关闭；无泵/关闭时逐位保持旧 PCFM 结果。

## 3. 论文 hybrid 对照模型

`src/network_nlo_eval/models/hybrid_raman.py`

```python
model = create_model(
    "hybrid_raman_paper", grid, fiber,
    raman_pumps=[RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )],
    include_isrs=True,
    fixed_point_iterations=24,
    spp_steps=32,
)
```

### 3.1 已复现（数值可验证）

- **论文 Eq. (2)**：信号与泵浦的耦合功率方程。任意数量 FW/BW 泵；耦合矩阵
  同时包含：
  - 泵浦-信号；
  - 泵浦-泵浦；
  - 信道间 ISRS（可用 `include_isrs=False` 关闭）；
  - 泵浦耗尽。
- **论文 Eq. (4)**：信号 ASE 的耦合方程，包含自发拉曼源项
  `2 h kappa B_i f_i`，并把上一跨 ASE 作为本跨初值。
- **每跨初值与跨间 ASE**：模型保存跨末 EDFA 后的 ASE，作为下一跨光纤入口的
  ASE 初值；路由层仍按信号传输因子搬运已累积 ASE。为避免重复计数，交给 EDFA
  的是本跨**新产生**的拉曼 ASE（已扣除输入 ASE 按信号净增益搬运的部分）。
- **FW/BW 统一处理**：FW 泵与信号同向积分；BW 泵从 `z=L` 反向积分，与正向
  信号做不动点迭代。无 BW 泵时只需一次前向积分；`0.2 W`、`c_57ch` 基准在
  18 次内收敛。
- **无泵退化**：直接走 PCFM 的 ISRS-only 路径（论文 Eq. (3) 的 LA 情形），
  与 PCFM 基线逐位一致（`tests/test_hybrid_raman.py`）。

### 3.2 仍是近似（明确边界）

1. **NLI 不是 Eq. (6)-(9)**。当前实现复用 PCFM 的 SCI/XCI 闭式核
   （`PCFMSpanKernel`），只把核里的归一化功率剖面 `p_i(z)` 换成 Eq. (2)/(4)
   ODE 解出的剖面。Raman 对 NLI 的作用通过剖面进入核，但没有论文的拟合系数
   `alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i` 与 Eq. (8)/(9) 的解析形式。
   因此 `ModelInfo.summary` 明确写 “NLI is approximate, not Eq. 6-9”。
2. 拉曼增益谱用 `RamanGainModel` 的 Gaussian/CZ 近似，不是论文 Fig. 4 的实测
   二氧化硅谱。
3. 泵浦 ASE 没有单独求解；信号 ASE 的耦合是完整的。泵浦 ASE 对信号 ASE 的
   贡献远小于泵浦-信号自发散射。
4. 不包含泵浦-信号 XPM、droop 的显式控制、动态增益控制、非均匀跨长拟合。
5. `physics_nlo_eval.RamanPump` 只作为增益形状对照；它本身也是未耗尽集总块，
   不能替代论文 coupled ODE。
6. EDFA ASE 用路由层既有 `NF h f B (G-1)` 口径；它与论文 Eq. (5) 的
   `2(G-1)n_sp h f B` 在高增益下近似等价，但不是逐项相同。Raman ASE 与
   EDFA ASE 在结果里是分开累积的。

## 4. 为什么 lumped 不能替代论文模型

集总等效做了三个会系统性改变结论的假设：

1. **泵浦未耗尽**：增益按 `P_p * L_eff` 线性外推。泵浦功率较高或信道数较多时，
   这会把 on-off 增益估高。
2. **增益集总在跨末**：跨内信号功率剖面不随拉曼增益变化，因此 NLI 完全不变。
   真实 FW 泵会抬高跨内功率、增加 NLI；BW 泵则把增益放在跨末，ASE 更高。
3. **ASE 用末端 `G-1` 近似**：不求解 Eq. (4) 的沿跨耦合，也不含跨间 ASE 初值
   对后续跨的影响。

对照基准（`c_57ch`、10×100 km、1450 nm / 0.2 W 后向泵）直接给出差异：

| 模型 | 平均 SNR (dB) | 平均 ASE (W) | 平均 NLI (W) | 冷启动 (ms) |
|---|---:|---:|---:|---:|
| `pcfm`（纯 EDFA） | 16.291 | 1.216e-5 | 2.911e-5 | 56 |
| `raman_lumped` | 17.221 | 4.194e-6 | 2.911e-5 | 57 |
| `hybrid_raman_paper` | 16.969 | 4.922e-6 | 3.042e-5 | 477 |

- lumped 的 NLI 与纯 EDFA **逐位相同**（2.911e-5），因为它不改跨内剖面；
  论文 ODE 的 NLI 高约 4.5%。
- lumped 的平均 SNR 比论文子集高 **0.25 dB**，主要来自增益高估与 NLI 低估。
- 论文子集的 ASE 比 lumped 高约 17%，因为它求解 Eq. (4) 的沿跨自发散射，
  而不是末端 `G-1` 近似。
- 同一泵浦下，论文 ODE 的 on-off 增益平均 **8.53 dB**，lumped 为 **10.25 dB**，
  物理 `RamanPump`（未耗尽）为 **9.60 dB**。泵浦耗尽把增益压低了约 1.7 dB。
- 小信号极限下，论文 ODE 的未耗尽增益与物理 `RamanPump` 逐信道差 **<0.1 dB**
  （`tests/test_hybrid_raman.py`），说明泵浦-信号耦合的方向与谱形实现正确；
  差异主要来自工作功率下的泵浦耗尽。

其余算例（`results/raman/paper_comparison.json`）：

| 算例 | 纯 EDFA | lumped | 论文子集 |
|---|---:|---:|---:|
| `c_57ch` 1 跨 | 26.665 | 27.688 | 27.436 |
| `scaling_40ch` 1 跨 | 27.837 | 30.827 | 30.437 |
| `scaling_40ch` 10 跨 | 17.675 | 20.511 | 20.123 |

所有数值均为平均 SNR (dB)，同一泵浦配置。论文子集始终低于 lumped，但仍高于
纯 EDFA；这与“Raman 降低 ASE、但增加 NLI”的定性结论一致。

## 5. 限制与后续

- 当前 NLI 仍是“PCFM 核 + ODE 剖面”的近似。若要 claim 论文 Eq. (6)-(9)，
  下一步需要实现 SPM/XPM 的 Raman 归一化闭式，并用 Eq. (10) 拟合
  `alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i`。
- 拉曼谱未标定到具体光纤；默认系数与物理 `RamanPump` 的 Gaussian 近似对齐，
  不是实测 G.652.D 参数。
- BW 泵不动点迭代在高损耗/强耗尽下可能需要更多次数；模型会返回
  `converged=False`，但当前公开接口只暴露路径结果，不暴露该标志。
- 未实现泵浦 ASE、泵浦-信号 XPM、droop 控制；跨长非均匀时 NLI 相干因子仍
  沿用 PCFM 的等跨近似。
- 模型内部的跨间 ASE 状态不经过 ROADM/booster；基准路径无 ROADM。带 ROADM 的
  链路仍可用，但 ODE 的 ASE 初值只跟踪跨末 EDFA 输出。
- 物理波形模型只做了增益形状对照；没有把波形仿真当作在线替代。

## 6. 复现命令

```bash
PYTHONPATH=src pytest tests/test_raman.py tests/test_hybrid_raman.py -q
PYTHONPATH=src pytest tests/test_pcfm.py -q
PYTHONPATH=src python scripts/raman_paper_comparison.py
```

结果：`results/raman/pump_sweep.json`、`results/raman/paper_comparison.json`。
