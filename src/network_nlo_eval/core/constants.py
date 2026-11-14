"""Physical constants and common conversion factors."""

import numpy as np
import scipy.constants as const

C_LIGHT: float = const.c  # Speed of light [m/s].
H_PLANCK: float = const.h  # Planck constant [J s].
PI: float = const.pi

# Conversion factors for power attenuation in decibels and nepers.
DB_TO_NP: float = np.log(10) / 10
NP_TO_DB: float = 10 / np.log(10)
