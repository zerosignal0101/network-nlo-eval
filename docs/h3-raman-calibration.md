# 实测拉曼谱标定：从测量数据到增益表与模型对照

日期：2026-11-09
工作树：`wt/h3-raman-calibration`（分支 `feat/h3-raman-calibration`，基线 `179d3e4`）

本文说明如何把一份**外部实测**的归一化拉曼增益谱接入仓库，得到可被
`RamanGainModel.from_file` 加载的 `RamanGainTable`，并与默认的
`gaussian`/`linear` 预设做形状、峰值和斜率对照。流程由
`scripts/raman_spectrum_calibration.py` 实现，测试见
`tests/test_raman_calibration_workflow.py`。

> **仓库内没有实测谱。** 本文和脚本都不内置、不下载任何测量数据。测试里的
> “谱”全部由测试代码合成，`source` 明确写为
> `synthetic fixture for tests, not measured`。任何标着 `measured` 的结论都必须
> 由调用方提供带来源的输入文件后才能复现。

## 1. 为什么需要这一步

H3 第十轮任务 T 已定位：无泵 S/L 边缘约 19 dB 的 ASE 差，根因是两种拉曼谱
（`RamanGainModel` 默认 Gaussian，峰值系数 6e-14 m/W；Semrau Table I(b) 的
线性代理，斜率 0.028 1/(W km THz)）在 21 THz 带宽上的 ISRS 倾斜不同，而不是
拉曼 ASE 源项记账错误。完整根因链见 `tests/test_h3_reference_anomaly.py` 和
`docs/raman-amplification.md`。

在那之前，仓库只有“表文件加载器”，没有“测量谱 → 标定表”的可复现入口：
标定字段容易和谱点分离，峰值/单调性/单位错误也只能到模型构造时才暴露。本
脚本把校验、provenance、预设对照和可选的单跨灵敏度放进同一条命令。

## 2. 输入格式

### 2.1 CSV

数据区固定两列，表头为 `shift_thz,normalized_gain`（也接受
`offsets_thz,normalized_gain`）。`# key = value` 头行携带 provenance：

```text
# fiber = SMF-28-like
# source = vendor report 2026-11, not bundled
# note = shift = |f_p - f_s|; normalized to the 13.2 THz peak
# temperature_k = 300.0
# peak_gain_coefficient_m_per_w = 6e-14
# reference_wavelength_nm = 1550.0
# reference_a_eff_um2 = 80.0
shift_thz,normalized_gain
0.0,0.0
1.0,0.08
13.2,1.0
```

### 2.2 JSON

最少需要 `offsets_thz`（或 `shift_thz`）与 `normalized_gain`；标定字段与 CSV
同名：

```json
{
  "offsets_thz": [0.0, 1.0, 13.2],
  "normalized_gain": [0.0, 0.08, 1.0],
  "fiber": "SMF-28-like",
  "source": "vendor report 2026-11, not bundled",
  "temperature_k": 300.0,
  "peak_gain_coefficient_m_per_w": 6e-14,
  "reference_wavelength_nm": 1550.0,
  "reference_a_eff_um2": 80.0
}
```

也可以直接把仓库的完整 `RamanGainTable` 文件（`format =
network_nlo_eval.raman_gain_table`）作为输入；此时温度、峰值系数、参考波长和
A_eff 从表里读取，脚本只做重新校验和对照。

### 2.3 硬性校验

`load_measured_spectrum` 在任何计算前拒绝：

| 检查 | 规则 |
|---|---|
| 维度/长度 | 一维、至少 2 点、频移与增益等长 |
| 有限性 | 所有值有限，拒绝 NaN/Inf |
| 频移 | 非负、**严格递增**（不自动排序） |
| 增益 | 非负，且至少一个正值 |
| 峰值 | 默认要求落在 `1 ± --peak-tolerance`（默认 0.02）内 |
| provenance | 必须有 `source`（命令行或输入文件） |

峰值超出容差时默认**拒绝**，不会静默缩放；只有显式 `--normalize-peak` 才
缩放，并把原始峰值和缩放比例写进报告及 `note`。

## 3. 运行

```bash
PYTHONPATH=src python scripts/raman_spectrum_calibration.py \
    --input measured.csv \
    --output results/raman/measured_gain_table.json \
    --report results/raman/measured_calibration.json \
    --source "vendor report 2026-11" \
    --fiber "SMF-28-like" \
    --temperature-k 300 \
    --peak-coefficient-m-per-w 6e-14 \
    --reference-wavelength-nm 1550 \
    --reference-a-eff-um2 80
```

默认只做标定和预设对照。可选的单跨灵敏度对照（明显更慢）：

```bash
PYTHONPATH=src python scripts/raman_spectrum_calibration.py \
    --input measured.csv \
    --output /tmp/measured_gain_table.json \
    --source "vendor report 2026-11" \
    --span-sensitivity both \
    --span-bands C --span-channels 40 --span-length-km 100 \
    --span-power-dbm 0 \
    --pump-wavelength-nm 1450 --pump-power-w 0.2
```

`--span-sensitivity` 取 `none`（默认）/`h3`/`numerical_dra`/`both`。灵敏度
算例固定为单跨、均匀栅格、单个后向泵浦；`--span-bands` 只决定网格中心，
实际覆盖范围由 `--span-channels` 与 100 GHz 间隔决定。参数通过命令行显式
给出，脚本不读 `data/`。

## 4. 输出一：RamanGainTable JSON

`--output` 写纯增益表，字段与 `docs/raman-amplification.md`、`raman_table.py`
一致：

| 字段 | 单位 | 含义 |
|---|---|---|
| `offsets_thz` | THz | 泵浦-信号绝对频移，非负严格递增 |
| `normalized_gain` | 1 | 归一化增益形；`1.0` 对应峰值系数 |
| `temperature_k` | K | 光纤温度；只影响 ASE 的声子因子 |
| `peak_gain_coefficient_m_per_w` | m/W | 峰值**局部**增益率，使用前除以 `A_eff` |
| `reference_wavelength_nm` | nm | 峰值系数的标定波长；**只溯源，不参与计算** |
| `reference_a_eff_um2` | µm² | 可选；标定时有效模面积，作为模型的参考 A_eff |
| `fiber`/`source`/`note` | 文本 | 溯源；`source` 必填 |

该文件可直接使用：

```python
from network_nlo_eval.models.raman import RamanGainModel

model = RamanGainModel.from_file("results/raman/measured_gain_table.json")
```

也可把路径传给支持表入口的模型（`pcfm` + `raman_model="lumped"`、
`pcfm_h3_raman`、`hybrid_raman_paper`/`RamanPaperSpanSolver`）：

```python
from network_nlo_eval.benchmarks.runner import make_model

model = make_model("pcfm_h3_raman", case, raman_gain_model="results/raman/measured_gain_table.json")
```

`buglia2024_ra_gn`、`zhang2024_dra_egn`、`numerical_dra`、`semrau2019` 和
`ecz` 的模型层仍只接受 `"gaussian"`/`"linear"` 名称；要用表谱做对照，请像
脚本一样直接构造 `RamanGainModel` 或对应的 span solver。

## 5. 输出二：对照报告

`--report`（默认 `results/raman/raman_spectrum_calibration.json`）记录：

| 块 | 内容 |
|---|---|
| `input` | 原始路径、SHA-256、点数、频移范围 |
| `table` | 输出路径、完整 `provenance`、标定字段、输入峰值、缩放比例、每个字段来自 `cli`/`input`/`default` |
| `comparison.grid` | 比较所用频移网格 |
| `comparison.reference` | 峰值系数、参考 A_eff、斜率/平均形状的参照带宽 |
| `comparison.shapes` | `measured`/`gaussian`/`linear` 的峰值、峰值频移、0–10 THz 割线斜率、21 THz 平均形状 |
| `comparison.deviation_vs_measured` | 预设相对实测表的 max/RMSE/mean 绝对偏差、相对 RMSE、相关系数 |
| `comparison.peak_shift_difference_thz` | 预设峰值频移减实测峰值频移 |
| `comparison.slope_ratio_vs_measured` | 预设斜率 / 实测斜率 |
| `span_sensitivity` | 可选；逐模型、逐谱的跨损/拉曼 ASE 摘要与 measured−gaussian 差值 |

斜率同时给出两种单位：

* `secant_slope_per_thz`：归一化形状的割线斜率，单位 1/THz；
* `secant_slope_1_per_w_km_thz`：乘
  `peak_gain_coefficient_m_per_w / A_eff * 1e3`，单位 1/(W km THz)，可与
  Semrau 的 0.028 1/(W km THz) 线性代理直接对照。

`mean_shape_0_21thz` 是 0–21 THz 上形状的带宽平均，正比于宽带 ISRS 倾斜的
一阶效应；表未覆盖的频移按 0 处理，所以它同时受测量范围影响，报告里的
`coverage_thz` 要一起看。

## 6. 把真实测量数据接进来

脚本只接受调用方给的数字，不能替代测量本身。接入时需要准备：

1. **频移轴**：把测量给出的泵浦-信号波长差换算成 `|f_p - f_s|`（THz），
   不要填绝对光频，也不要填带符号的频差；表只接受非负、严格递增的频移。
2. **归一化增益**：把测量谱除以自己的峰值，得到无量纲形状。若原始数据是
   on-off dB、1/(km W) 或 m/W，先换算成形状；脚本不会猜测单位。峰值明显
   不为 1 时会被拒绝，确认无误后再用 `--normalize-peak`。
3. **峰值系数**：`peak_gain_coefficient_m_per_w` 是峰值**局部**增益率
   （泵浦未耗尽、单位泵浦功率、单位长度），不是跨段 on-off 增益，也不是
   `g_R/A_eff`。它必须来自同一套测量/厂商数据；脚本在缺失时回落到模型默认
   6e-14 m/W，并在 `note` 和报告的 `provenance_sources` 里标为 `default`，
   **不能**把这种回落当作实测标定。
4. **参考波长**：填写峰值系数的标定波长，只用于溯源。模型不会按信号波长
   对它做缩放；跨 C+L/S 使用时绝对增益偏差需要另行评估。
5. **A_eff**：填写标定时的有效模面积。模型在调用方没有显式给 `A_eff` 时
   使用它；逐信道 `A_eff` 仍可由模型调用参数覆盖。
6. **温度**：只影响 ASE 的声子占据因子，不改变 on-off 增益；不要用它吸收
   谱形误差。
7. **来源**：`--source` 必填，建议同时写 `--fiber`、测量日期、仪器/方法、
   归一化频移和任何平滑/去噪处理；这些会原样进入表和报告。

拿到表后先在目标跨段上做一次 sanity check：用 `--span-sensitivity both`
看跨损和拉曼 ASE 相对默认 Gaussian 的变化方向与量级是否合理，再进入完整
网络评估。

## 7. 已知限制

* 脚本不做测量不确定度传播；误差棒、重复测量和温度漂移需要调用方另行
  建模。
* 表在 `offsets_hz` 范围外按 0 处理（与 `RamanGainModel` 一致），不会外推。
  测量范围不足时，超出部分会被静默截断为无拉曼增益；报告里的
  `coverage_thz` 和单跨灵敏度的覆盖检查会提示这一点。
* `gaussian`/`linear` 是模型预设，不是任何具体光纤的标定；对照只回答“实测
  谱离默认预设多远”，不回答“哪个预设更正确”。
* 单跨灵敏度只比较 `pcfm_h3_raman` 与 `numerical_dra` 的跨损/拉曼 ASE，
  不含 NLI、EDFA ASE 或跨间相干；它是标定 sanity check，不是精度矩阵。
* 默认路径（不传表、不启用 opt-in）逐位不变；本流程没有修改
  `RamanGainModel` 的默认参数。

## 8. 复现

```bash
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest -q \
    tests/test_raman_calibration_workflow.py \
    tests/test_raman_table.py tests/test_raman.py tests/test_h3_reference_anomaly.py
ruff check scripts/raman_spectrum_calibration.py tests/test_raman_calibration_workflow.py
```

测试只使用合成夹具，不访问网络，不读取 `data/`。
