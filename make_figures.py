# makes all figures for the report (not part of the package, do not commit)

import os

import matplotlib.pyplot as plt
import numpy as np

from bounce.analysis import (
    check_bounces,
    plot_results,
    plot_statistical_error,
    statistical_error_study,
)
from bounce.simulation import run_simulation

os.makedirs("figures", exist_ok=True)

# 1. the 3 pictures of one run (Kevin)
results = run_simulation(n=10000, seed=1)
plot_results(results, "figures")

# 2. the P(k) check (Kevin)
print("check bounces:", check_bounces())

# 3. the statistical error with N = 1000, 10000, 100000 (Kevin)
rows, slope = statistical_error_study()
print("statistical error:", rows)
print("slope:", slope)
plot_statistical_error(rows, "figures")

# 4. specular vs diffuse, with and without drag
runs = {
    "specular, no drag": dict(p_specular=1.0, gamma=0.0),
    "diffuse, no drag": dict(p_specular=0.0, gamma=0.0),
    "specular, drag 0.3": dict(p_specular=1.0, gamma=0.3),
    "diffuse, drag 0.3": dict(p_specular=0.0, gamma=0.3),
}
fig, ax = plt.subplots()
for label, settings in runs.items():
    res = run_simulation(n=10000, seed=0, **settings)
    r = res["final_r"]
    print(f"{label:20s} mean r = {np.mean(r):.3f}")
    ax.hist(r, bins=80, range=(0, 60), density=True, histtype="step", label=label)
ax.set_yscale("log")
ax.set_xlabel("final distance r from the source")
ax.set_ylabel("probability density")
ax.set_title("Specular vs diffuse, with and without drag")
ax.legend()
fig.savefig("figures/comparisons.png", dpi=150)
plt.close(fig)

# 5. histograms of the random speeds and angles
from bounce.sampling import cosine_directions, maxwell_speeds

rng = np.random.default_rng(1)
v = maxwell_speeds(100000, 1.0, rng)
x = np.linspace(0, 5, 200)
f = np.sqrt(2 / np.pi) * x**2 * np.exp(-(x**2) / 2)
fig, ax = plt.subplots()
ax.hist(v, bins=60, density=True, label="samples")
ax.plot(x, f, "r", label="Maxwell-Boltzmann f(v)")
ax.set_xlabel("speed v / sigma")
ax.set_ylabel("probability density")
ax.legend()
fig.savefig("figures/speeds.png", dpi=150)
plt.close(fig)

d = cosine_directions(100000, rng)
theta = np.arccos(d[:, 2])
t = np.linspace(0, np.pi / 2, 200)
fig, ax = plt.subplots()
ax.hist(theta, bins=60, density=True, label="samples")
ax.plot(t, np.sin(2 * t), "r", label="cosine law sin(2 theta)")
ax.set_xlabel("theta (angle from the vertical)")
ax.set_ylabel("probability density")
ax.legend()
fig.savefig("figures/angles.png", dpi=150)
plt.close(fig)
