"""网络中的物理元件配置模型，使用 Pydantic."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_nlo_eval.core.types import NDArrayFloat


class FiberSpanConfig(BaseModel):
    """单跨段光纤的静态配置参数。

    所有参数均为参考波长下的典型值，模型将根据实际波长进行修正。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    length_km: float = Field(..., gt=0, description="光纤链路长度 (km)")
    max_span_length_km: float = Field(100.0, gt=0, description="放大跨段最大长度 (km)")

    # 衰减
    attenuation_db_km_ref: float = Field(0.2, gt=0, description="参考波长下的衰减系数 (dB/km)")
    use_attenuation_polynomial: bool = Field(False, description="是否使用 alpha0/1/2 波长衰减曲线")
    # Polynomial defaults follow the original GSNR calculator reference.
    attenuation_alpha2: float = Field(3.7685e-6, description="衰减曲线的二次项系数")
    attenuation_alpha1: float = Field(-7.3764e-5, description="衰减曲线的一次项系数")
    attenuation_alpha0: float = Field(0.162, description="衰减曲线的常数项")
    reference_wavelength_nm: float = Field(1550.0, description="衰减曲线的参考波长 (nm)")

    # 色散
    dispersion_parameter_d: float = Field(17.0, description="参考波长下的色散参数 D (ps/(nm*km))")
    dispersion_slope_s: float = Field(0.06, description="参考波长下的色散斜率 S (ps/(nm^2*km))")

    # 非线性
    nonlinear_index_n2: float = Field(2.6e-20, ge=0, description="光纤的非线性折射率 n2 (m^2/W)")
    effective_area_um2_ref: float = Field(80.0, gt=0, description="参考波长下的有效模面积 A_eff (um^2)")
    aeff_slope_um2_nm: float = Field(0.05, description="有效模面积随波长变化的斜率 (um^2/nm)")
    raman_pumps: list["RamanPumpConfig"] = Field(default_factory=list, description="该链路的分布式拉曼泵")


class EDFAConfig(BaseModel):
    """掺铒光纤放大器 (EDFA) 的配置参数."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_gain_db: float | None = Field(None, ge=0, description="固定增益；None 表示自动补偿跨段损耗")
    noise_figure_db: float | NDArrayFloat = Field(5.0, description="放大器噪声系数 (NF) (dB)")
    # 可选：增益平坦度/纹波 (未来可扩展为NDArrayFloat)
    gain_ripple_db: NDArrayFloat | None = Field(None, description="增益纹波 (dB), 每个信道一个值")

    @model_validator(mode="after")
    def validate_noise_figure(self) -> "EDFAConfig":
        """噪声系数和纹波必须为有限值，NF 不得为负."""
        noise_figure = np.asarray(self.noise_figure_db)
        if np.any(~np.isfinite(noise_figure)) or np.any(noise_figure < 0):
            raise ValueError("EDFA noise figure must contain finite non-negative values.")
        if self.gain_ripple_db is not None and np.any(~np.isfinite(self.gain_ripple_db)):
            raise ValueError("EDFA gain ripple must contain finite values.")
        return self


class RamanPumpConfig(BaseModel):
    """分布式拉曼泵配置，与物理波形模型的 PumpSpec 一一对应."""

    wavelength_nm: float = Field(..., gt=0, description="泵浦波长 (nm)")
    power_w: float = Field(..., ge=0, description="泵浦功率 (W)")
    direction: Literal["forward", "backward"] = Field("backward", description="泵浦方向")
    attenuation_db_per_km: float = Field(0.25, ge=0, description="泵浦衰减 (dB/km)")


class ROADMConfig(BaseModel):
    """可重构光分插复用器的损耗、WSS 传递函数和放大配置."""

    insertion_loss_db: float = Field(5.0, ge=0, description="WSS/连接器插入损耗 (dB)")
    filtering_penalty_db: float = Field(0.2, ge=0, description="统计模型中的级联滤波 SNR 代价 (dB)")
    passband_hz: float | None = Field(None, gt=0, description="WSS 3-dB 通带；None 使用信道间隔")
    filter_order: int = Field(4, ge=1, description="超高斯 WSS 阶数")
    express_attenuation_db: float = Field(0.0, ge=0, description="直通端口附加衰减 (dB)")
    add_drop_loss_db: float = Field(1.0, ge=0, description="本地 add/drop 端口附加损耗 (dB)")
    booster_noise_figure_db: float = Field(5.0, ge=0, description="补偿 ROADM 损耗的 booster NF (dB)")
    equalize_output_power: bool = Field(True, description="是否由 booster 恢复每信道入纤功率")
    blocked_channels: list[int] = Field(default_factory=list, description="静态阻断的信道索引")

    @model_validator(mode="after")
    def validate_blocked_channels(self) -> "ROADMConfig":
        """拒绝重复或负信道索引."""
        if any(index < 0 for index in self.blocked_channels):
            raise ValueError("ROADM blocked channel indices must be non-negative.")
        if len(set(self.blocked_channels)) != len(self.blocked_channels):
            raise ValueError("ROADM blocked channel indices must be unique.")
        return self

    def path_loss_db(self, operation: Literal["express", "add", "drop"] = "express") -> float:
        """返回给定交换操作的确定性损耗."""
        port_loss = self.express_attenuation_db if operation == "express" else self.add_drop_loss_db
        return self.insertion_loss_db + port_loss
