"""Example Gallery Plot.

====================
This is a simple plot to test Sphinx-Gallery.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
