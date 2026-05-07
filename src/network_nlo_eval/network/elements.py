"""网络中的物理元件配置模型，使用 Pydantic."""

from pydantic import BaseModel, Field

from network_nlo_eval.core.types import NDArrayFloat


class FiberSpanConfig(BaseModel):
    """单跨段光纤的静态配置参数。

    所有参数均为参考波长下的典型值，模型将根据实际波长进行修正。
    """

    length_km: float = Field(..., description="光纤跨段长度 (km)")

    # 衰减
    attenuation_db_km_ref: float = Field(0.2, description="参考波长下的衰减系数 (dB/km)")
    attenuation_alpha2: float = Field(3.7685e-6, description="衰减曲线的二次项系数")  # From Ref. [1] in gsnr_calculator
    attenuation_alpha1: float = Field(-7.3764e-5, description="衰减曲线的一次项系数")
    attenuation_alpha0: float = Field(0.162, description="衰减曲线的常数项")
    reference_wavelength_nm: float = Field(1550.0, description="衰减曲线的参考波长 (nm)")

    # 色散
    dispersion_parameter_d: float = Field(17.0, description="参考波长下的色散参数 D (ps/(nm*km))")
    dispersion_slope_s: float = Field(0.06, description="参考波长下的色散斜率 S (ps/(nm^2*km))")

    # 非线性
    nonlinear_index_n2: float = Field(2.6e-20, description="光纤的非线性折射率 n2 (m^2/W)")
    effective_area_um2_ref: float = Field(80.0, description="参考波长下的有效模面积 A_eff (um^2)")
    aeff_slope_um2_nm: float = Field(0.05, description="有效模面积随波长变化的斜率 (um^2/nm)")

    class Config:
        arbitrary_types_allowed = True  # 允许模型中使用 numpy 数组等任意类型


class EDFAConfig(BaseModel):
    """掺铒光纤放大器 (EDFA) 的配置参数."""

    target_gain_db: float = Field(20.0, description="放大器目标增益 (dB)")
    noise_figure_db: float | NDArrayFloat = Field(5.0, description="放大器噪声系数 (NF) (dB)")
    # 可选：增益平坦度/纹波 (未来可扩展为NDArrayFloat)
    gain_ripple_db: NDArrayFloat | None = Field(None, description="增益纹波 (dB), 每个信道一个值")

    class Config:
        arbitrary_types_allowed = True


class ROADMConfig(BaseModel):
    """可重构光分插复用器 (ROADM) 的配置参数."""

    insertion_loss_db: float = Field(12.0, description="ROADM 的插入损耗 (dB)")
    filtering_penalty_db: float = Field(0.5, description="ROADM 滤波引入的额外损耗 (dB)")
