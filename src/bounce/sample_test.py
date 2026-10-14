# PERSONAL TEST FOR REJECTION SAMPLING CODE

# NOTE: this is NOT the file that will be used for the pytest
# tests. This is for me to make sure that the script works

import numpy as np
import random
import sampling as samp
import matplotlib.pyplot as plt

rng = np.random.default_rng()
N = 1000
sigma = 0.1

const = np.sqrt(2 * np.pi) * sigma**(-3)
x = np.linspace(0, 1, 500)
y = const * (x ** 2) * np.exp(-(x ** 2) / (2 * sigma**2))

particles, sample_data = samp.maxwell_speeds(N, sigma, rng)
particles = particles.tolist()
v, fv = zip(*particles)

fig, ax = plt.subplots()
ax.scatter(v, fv)
ax.plot(x, y)
#ax.hist(fv, bins = 50, density = True)
plt.show()