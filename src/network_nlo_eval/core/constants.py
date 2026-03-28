"""物理常数与常用转换因子."""

import numpy as np
import scipy.constants as const

C_LIGHT: float = const.c  # 光速 [m/s]
H_PLANCK: float = const.h  # 普朗克常数 [J·s]
PI: float = const.pi  # 圆周率

# 常用转换
DB_TO_NP: float = np.log(10) / 10  # dB 到 Neper/m 的转换因子 (损耗)
NP_TO_DB: float = 10 / np.log(10)  # Neper/m 到 dB 的转换因子 (损耗)
