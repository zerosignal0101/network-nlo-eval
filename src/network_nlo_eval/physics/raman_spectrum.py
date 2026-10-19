r"""宽波段硅基（熔融石英）受激拉曼增益谱 — FG-U1 `tabular` 档的数据与插值。

为什么需要这个模块
------------------
`numba_kernels` 里 legacy 档的拉曼耦合是一条**合成折线**::

    df <= 13.2 THz :  g = cr * df                  # 线性爬升
    13.2..15.6 THz :  g = peak * (15.6-df)/2.4     # 线性跌到零
    df >= 15.6 THz :  g = 0                        # 恒零

这对 C+L（~11.45 THz 跨距）尚可，但对 S+C+L+U（~27 THz 跨距）有两个硬伤：
`>15.6 THz` 被**人为置零**，而真实石英谱在 20–30 THz 是**小而非零**；
13.2 THz 附近是**宽峰**而不是线性爬升。本模块提供一条有出处的宽波段谱替换它。

出处与推导（不是"凭印象编的曲线"）
-----------------------------------
谱形来自**已发表且参数完整的**熔融石英拉曼响应函数，而不是从图上 digitization
出来的散点：

* **响应函数**（Lin & Agrawal, Opt. Lett. **31**, 3086 (2006),
  doi:10.1364/JOSAB.6.001159，"Raman response function for silica fibers"）::

      h_R(t) = (tau1^2 + tau2^2) / (tau1 * tau2^2) * exp(-t/tau2) * sin(t/tau1)

  石英的拟合参数为 **tau1 = 12.2 fs、tau2 = 32 fs**（f_R = 0.18）。

* **从响应函数到功率增益谱**：功率增益系数正比于拉曼响应函数傅里叶变换的虚部
  （Agrawal, *Nonlinear Fiber Optics* Eq. (2.3.42)；Hollenbeck & Cantrell,
  J. Opt. Soc. Am. B **19**, 2886 (2002) Eq. (7)）::

      g_R(w)  ∝  Im{ h_R~(w) },   h_R~(w) = ∫_0^∞ h_R(t) exp(-i w t) dt

  对上面的阻尼谐振子做这个变换（令 ``b = 1/tau1``、``g = 1/tau2``、``w`` 为角频率）
  可以**解析地**化简成一个闭式表达式::

      -Im{ h_R~(w) } = 2 b g w / ( (b^2 + g^2 - w^2)^2 + (2 g w)^2 )

  这个闭式就是本模块 `tabular` 档的谱源。它给出的峰位是 **13.08 THz**
  （文献常引的 13.2 THz；两者相差来自 tau1/tau2 是四位有效数字的拟合值，
  详见 `docs/fg-u1_raman_spectrum.md`）。闭式与对 h_R(t) 的直接数值傅里叶变换
  互校一致到 ~1e-9（见 `tests/test_fg_u1_raman_spectrum.py`）。

单位与归一化（与 legacy 档的对应关系）
--------------------------------------
`cr_w_m_hz` 的既有语义是"**斜率**"，使得 ``G(13.2 THz) = cr * 13.2e12``。为了让
两个档位只差**谱形**、不差**幅度**（否则 legacy/tabular 的比较会被幅度混淆），
`tabular` 档用同一锚点归一化::

    legacy : G_legacy(df) = cr_w_m_hz * (df * 1e12)
    tabular: G_tabular(df) = cr_w_m_hz * 13.2e12 * S(df)

其中 ``S`` 是本模块的**无量纲**形状函数，按构造 ``S(13.2 THz) ≡ 1``。于是

* 两档在 13.2 THz 处**完全相等**（逐位相等的锚点）；
* ``cr_w_m_hz`` 在两档中的**单位与含义完全相同**（1/(W·m·Hz) 的斜率）；
* 差别纯粹来自 ``S`` 相对 legacy 折线的偏离。

注意 `cr_w_m_hz` 标注的单位 ``1/(W·m·Hz)`` 与文献里 ``g_R`` 的 ``m/W``
（每单位*频率*）之间其实差一个长度量纲——这是 legacy 档沿用下来的既有约定，
**本模块不做改动**（改动会破坏 FG-4a 的金标准）。本模块只保证 `tabular` 档
与 legacy 档在这同一个约定下语义一致。

超出表范围的截断
----------------
表覆盖 0 ~ 40 THz（超过 brief 要求的 30 THz 留了余量）。S+C+L+U 的最大
跨距约 27 THz，落在表内。``df > 40 THz`` 的谱值被**显式置 0**，被截掉的尾巴
在 40 THz 处只有峰值的 ~2.5%，量级见文档。
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np

from network_nlo_eval.core.constants import H_PLANCK
from network_nlo_eval.core.types import NDArrayFloat

# --- 谱源参数（全部来自上文的出处，见模块 docstring）----------------------

#: Lin & Agrawal (2006) 的石英拟合参数 tau1 (s) —— 分子振动的"振荡"时间。
TAU1_S = 12.2e-15

#: Lin & Agrawal (2006) 的石英拟合参数 tau2 (s) —— 分子振动的"衰减"时间。
TAU2_S = 32.0e-15

#: 归一化锚点频差 (THz)：两档在此处**逐位相等**，故 legacy 的 13.2/15.6
#: 折点常量与真实谱的峰位对齐关系在这里被保留下来。
PEAK_ANCHOR_THZ = 13.2

#: 表格覆盖的频率上限 (THz)。S+C+L+U 的最大跨距 ~27 THz，留了余量。
TABLE_MAX_THZ = 40.0

#: 表格步长 (THz)。均匀栅格是为了让 njit 里的插值退化成一次索引 + 一次线性混合。
TABLE_STEP_THZ = 0.05

#: 玻尔兹曼常数 (J/K)，只用于玻色占据数因子。
BOLTZMANN_J_PER_K = 1.380649e-23

#: 平面波角频率 ↔ Hz 的换算因子。
_TWO_PI = 2.0 * np.pi

_THZ = 1e12

#: 参考光纤温度 (K)，只影响"不对称"档的玻色因子。不对称档未在
#: `FiberSpanConfig` 上暴露温度，因此这里是一个**显式默认值**（见文档"未测项"）。
DEFAULT_TEMPERATURE_K = 300.0

#: 采样密度与有效范围的机器可读描述，写进证据 JSON 以便复算。
SPECTRUM_METADATA: dict[str, object] = {
    "material": "fused silica (SiO2) single-mode fibre",
    "response_function": "h_R(t) = (tau1^2+tau2^2)/(tau1*tau2^2) * exp(-t/tau2) * sin(t/tau1)",
    "primary_reference": (
        "Q. Lin and G. P. Agrawal, 'Raman response function for silica fibers', "
        "Optics Letters 31(19), 3086 (2006), doi:10.1364/JOSAB.6.001159"
    ),
    "gain_derivation": (
        "g_R(w) proportional to Im{FT[h_R(t)]} "
        "(G. P. Agrawal, Nonlinear Fiber Optics, Eq. (2.3.42); "
        "D. Hollenbeck and C. Cantrell, J. Opt. Soc. Am. B 19(12), 2886 (2002), Eq. (7))"
    ),
    "closed_form": "-Im{h_R~(w)} = 2*b*g*w / ((b^2 + g^2 - w^2)^2 + (2*g*w)^2), b=1/tau1, g=1/tau2",
    "tau1_s": TAU1_S,
    "tau2_s": TAU2_S,
    "normalisation_anchor_thz": PEAK_ANCHOR_THZ,
    "table_max_thz": TABLE_MAX_THZ,
    "table_step_thz": TABLE_STEP_THZ,
}

#: 提交进仓库的数据表（人可读、可 diff、可被第三方独立复算）。
TABLE_CSV_PATH = Path(__file__).resolve().parents[3] / "references" / "raman_gain_silica_lin_agrawal_2006.csv"


def raman_gain_spectrum_closed_form(df_thz: NDArrayFloat | float) -> NDArrayFloat:
    """闭式功率增益谱 ``-Im{FT[h_R(t)]}``，**未归一化**。

    这是 `tabular` 档的唯一谱源；表 (:data:`TABLE_CSV_PATH`) 是它的采样，
    两者由 ``tests/test_fg_u1_raman_spectrum.py`` 互校。

    Args:
        df_thz: 频差，单位 THz。允许标量或数组。

    Returns
    -------
        NDArrayFloat: 与输入同形的 float64 数组/标量，量级为 1/(m·W) 的未归一化增益。
    """
    b = 1.0 / TAU1_S
    g = 1.0 / TAU2_S
    w = _TWO_PI * np.asarray(df_thz, dtype=np.float64) * _THZ
    denom = (b * b + g * g - w * w) ** 2 + (2.0 * g * w) ** 2
    return 2.0 * b * g * w / denom


def raman_gain_shape(df_thz: NDArrayFloat | float) -> NDArrayFloat:
    """归一化形状函数 ``S(df)``，按构造 ``S(PEAK_ANCHOR_THZ) ≡ 1``。

    ``tabular`` 档的 G 矩阵按 ``G = cr_w_m_hz * 13.2e12 * S(df)`` 组装，
    因此形状与幅度的职责是分开的：`cr_w_m_hz` 管幅度，`S` 管谱形。

    Args:
        df_thz: 频差，单位 THz。

    Returns
    -------
        NDArrayFloat: 无量纲形状值，``S(13.2 THz) == 1``。
    """
    return raman_gain_spectrum_closed_form(df_thz) / raman_gain_spectrum_closed_form(PEAK_ANCHOR_THZ)


def bose_occupation(df_thz: NDArrayFloat | float, temperature_k: float = DEFAULT_TEMPERATURE_K) -> NDArrayFloat:
    """玻色-爱因斯坦占据数 ``n_B(w) = 1 / (exp(hw/kT) - 1)``。

    出处：玻色-爱因斯坦分布，Landau & Lifshitz, *Statistical Physics* Part 1,
    §"光子气体"。这里用 ``h``（普朗克常数）而不是 ``ħ``，因为乘的是**普通频率**
    ``f = w / 2π``，故指数为 ``h·f / (k_B T)``。

    Args:
        df_thz: 频差，单位 THz。
        temperature_k: 参考温度 (K)。

    Returns
    -------
        NDArrayFloat: 无量纲占据数；``df -> 0`` 时发散（物理上正确）。
    """
    if temperature_k <= 0.0:
        raise ValueError(f"temperature_k must be > 0, got {temperature_k}.")
    f_hz = np.asarray(df_thz, dtype=np.float64) * _THZ
    exponent = (H_PLANCK * f_hz) / (BOLTZMANN_J_PER_K * temperature_k)
    with np.errstate(divide="ignore", over="ignore"):
        return 1.0 / np.expm1(exponent)


def asymmetry_factor(
    df_thz: NDArrayFloat | float, side: str, temperature_k: float = DEFAULT_TEMPERATURE_K
) -> NDArrayFloat:
    """Stokes / anti-Stokes 不对称因子（相对 ``symmetric`` 近似）。

    物理依据：处于温度 ``T`` 热平衡的振子群中，**Stokes** 过程受激放大对应
    ``n_B + 1`` 个可湮灭的振动量子，**anti-Stokes** 过程对应 ``n_B`` 个可被吸收
    的振动量子。因此同一 ``|df|`` 上两侧的净增益并不相等，比值就是 ``n_B``。

    常用的**对称近似**（也就是 legacy 档 `abs(f_i - f_k)` 所隐含的）相当于把两侧
    都取两者的平均 ``n_B + 1/2``。本函数返回相对该对称近似的比值::

        Stokes    : (n_B + 1) / (n_B + 1/2)
        anti-Stokes: n_B / (n_B + 1/2)

    Args:
        df_thz: 频差（取绝对值使用），单位 THz。
        side: ``"stokes"`` 或 ``"anti_stokes"``。
        temperature_k: 参考温度 (K)。

    Returns
    -------
        NDArrayFloat: 无量纲因子；对称档等价于乘 1.0。
    """
    if side not in ("stokes", "anti_stokes"):
        raise ValueError(f'side must be "stokes" or "anti_stokes", got {side!r}.')
    n_b = bose_occupation(np.abs(np.asarray(df_thz, dtype=np.float64)), temperature_k)
    sym = n_b + 0.5
    return (n_b + 1.0) / sym if side == "stokes" else n_b / sym


def spectrum_table_df_thz() -> NDArrayFloat:
    """表格的频差栅格 (THz)，均匀、含 0 与 :data:`TABLE_MAX_THZ` 两端。"""
    n = int(round(TABLE_MAX_THZ / TABLE_STEP_THZ)) + 1
    return np.arange(n, dtype=np.float64) * TABLE_STEP_THZ


def spectrum_table_shape() -> NDArrayFloat:
    """表格的形状值 ``S``，与 :func:`spectrum_table_df_thz` 逐点对应。"""
    return np.asarray(raman_gain_shape(spectrum_table_df_thz()), dtype=np.float64)


def write_spectrum_csv(path: Path | None = None) -> Path:
    """把闭式谱写成提交用的 CSV（数据 + 出处表头）。

    这是**证据生成器**，不是测试。仓库里提交的那份就是它生成的，
    ``tests/test_fg_u1_raman_spectrum.py`` 会证明 CSV 与闭式一致。

    Args:
        path: 输出路径，默认 :data:`TABLE_CSV_PATH`。

    Returns
    -------
        Path: 实际写出的路径。
    """
    target = Path(path) if path is not None else TABLE_CSV_PATH
    df_thz = spectrum_table_df_thz()
    shape = spectrum_table_shape()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([f"# {key}: {value}" for key, value in SPECTRUM_METADATA.items()])
        writer.writerow(["df_thz", "shape_s"])
        for df_value, shape_value in zip(df_thz, shape, strict=True):
            writer.writerow([repr(float(df_value)), repr(float(shape_value))])
    return target


@functools.lru_cache(maxsize=1)
def load_raman_gain_table() -> tuple[NDArrayFloat, NDArrayFloat]:
    """加载提交在仓库里的谱表，返回 ``(df_thz, shape_s)``。

    刻意**从磁盘读**而不是从闭式现算：这样"用了哪条曲线"就是一件可以被 diff、
    被第三方独立复核的事情，而不是埋在代码里的一段计算。

    Returns
    -------
        tuple: ``(df_thz, shape_s)`` 两个等长 float64 数组。

    Raises
    ------
        FileNotFoundError: 谱表 CSV 不存在（提示运行 `write_spectrum_csv()`）。
    """
    if not TABLE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Raman spectrum table not found at {TABLE_CSV_PATH}; "
            "regenerate it with network_nlo_eval.physics.raman_spectrum.write_spectrum_csv()."
        )
    df_list: list[float] = []
    shape_list: list[float] = []
    with TABLE_CSV_PATH.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "df_thz":
                continue
            df_list.append(float(row[0]))
            shape_list.append(float(row[1]))
    return np.array(df_list, dtype=np.float64), np.array(shape_list, dtype=np.float64)


def interp_raman_gain_shape(
    df_thz: NDArrayFloat | float, df_grid: NDArrayFloat, shape_grid: NDArrayFloat
) -> NDArrayFloat:
    """在谱表上做线性插值（纯 numpy 版，供自检与脚本使用）。

    核内走的是 :func:`_interp_uniform_shape`（njit），两者必须给出相同结果。

    超出 ``[0, TABLE_MAX_THZ]`` 的频差返回 0：真实谱在 40 THz 处只剩峰值的
    ~2.5%，且 Lorentzian 尾巴本就趋于 0，这里选择显式截断而不是外推。

    Args:
        df_thz: 待求值的频差 (THz)。
        df_grid: 严格递增的频差栅格 (THz)。
        shape_grid: 与 ``df_grid`` 对应的形状值。

    Returns
    -------
        NDArrayFloat: 插值得到的形状值。
    """
    query = np.asarray(df_thz, dtype=np.float64)
    clipped = np.clip(query, df_grid[0], df_grid[-1])
    out = np.interp(clipped, df_grid, shape_grid)
    return np.where((query < df_grid[0]) | (query > df_grid[-1]), 0.0, out)


def _interp_uniform_shape(df_thz: float, df0: float, step: float, n: int, shape_values: NDArrayFloat) -> float:
    """Njit 版均匀栅格线性插值——与 :func:`interp_raman_gain_shape` 必须一致。

    热路径（RK4 / matvec）**不**调用本函数；它只在构建期生成 G 矩阵时被调用一次，
    因此这里允许有分支。
    """
    if df_thz < df0 or df_thz > df0 + step * (n - 1):
        return 0.0
    x = (df_thz - df0) / step
    idx = int(x)
    if idx >= n - 1:
        idx = n - 2
    frac = x - idx
    return shape_values[idx] * (1.0 - frac) + shape_values[idx + 1] * frac
