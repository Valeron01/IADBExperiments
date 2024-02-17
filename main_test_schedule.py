import numpy as np
from matplotlib import pyplot as plt


x = np.linspace(0, 1, 1000)
sched = -np.cos(x * np.pi / 2) + 1

plt.plot(x, sched)
plt.show()