# Theory formulas and validation checks (Javad). The plots at the end are
# Kevin's part.

import numpy as np
from scipy import stats

from .particles import Particles
from .simulation import run_simulation


# Theory formulas



def range_no_drag(v, g=1.0):
    # range of one flight without drag that starts and ends on the floor:
    # R = v^2 sin(2 theta) / g. With v_h = v sin(theta) and v_z = v cos(theta)
    # this is the same as 2 v_h v_z / g
    v_h = np.hypot(v[:, 0], v[:, 1])
    return 2 * v_h * v[:, 2] / g


def mean_range(sigma=1.0, g=1.0):
    # mean range <R> = <v^2> <sin(2 theta)> / g. Speed and direction are
    # independent, so we can average them alone:
    #     Maxwell-Boltzmann:              <v^2> = 3 sigma^2
    #     cosine law p = sin(2 theta):    <sin(2 theta)> = pi / 4
    # so <R> = 3 pi sigma^2 / (4 g)
    return 3 * np.pi * sigma**2 / (4 * g)


def mean_range_squared(sigma=1.0, g=1.0):
    # mean of R^2 = <v^4> <sin^2(2 theta)> / g^2
    #     Maxwell-Boltzmann:   <v^4> = 15 sigma^4
    #     cosine law:          <sin^2(2 theta)> = 2 / 3
    # so <R^2> = 10 sigma^4 / g^2. This is the size of one step of the
    # random walk with diffuse bounces
    return 10 * sigma**4 / g**2


def bounce_probability(k, p_stick):
    # probability of exactly k bounces. Every landing is like a coin flip:
    # k times "bounce" (1 - p_stick) and then one time "stick" (p_stick)
    return (1 - p_stick) ** np.asarray(k) * p_stick


def mean_bounces(p_stick):
    # mean of P(k): the sum of k (1 - p)^k p is (1 - p) / p
    return (1 - p_stick) / p_stick


def terminal_velocity(m=1.0, g=1.0, gamma=1.0):
    # with linear drag the particle stops speeding up when gamma v = m g,
    # so the terminal velocity is v_t = m g / gamma
    return m * g / gamma


def max_energy_loss(dt, g=1.0, m=1.0):
    # Without drag the energy can only change at a landing, because of the
    # linear interpolation. The straight line between the old and the new
    # point is under the parabola, so at the interpolated time the particle is
    # really a little above the floor: at most g dt^2 / 8 (in the middle of the
    # step). We put it on the floor, so it loses at most m g * g dt^2 / 8.
    return m * g**2 * dt**2 / 8



# Validation checks

# Every check gives back a list of dicts with "name", "value", "target"
# (as text) and "ok" (True or False).


def check_range(n=20000, dt=0.01, seed=0):
    # first landing without drag compared with R = v^2 sin(2 theta) / g
    res = run_simulation(n=n, dt=dt, p_stick=1.0, seed=seed)
    error = np.abs(res["first_r"] - range_no_drag(res["v0"]))
    # we only see the landing at the end of a step, so the error can not be
    # more than the distance the particle moves in one step, v_h dt
    v_h = np.hypot(res["v0"][:, 0], res["v0"][:, 1])
    worst = np.max(error / (v_h * dt))

    # the mean range of all particles compared with 3 pi sigma^2 / (4 g).
    # It is a mean of n random values, so its error is std / sqrt(n)
    mean = np.mean(res["first_r"])
    err = np.std(res["first_r"]) / np.sqrt(n)
    theory = mean_range()
    return [
        {
            "name": "first landing: max error / (v_h dt)",
            "value": worst,
            "target": "<= 1",
            "ok": worst <= 1,
        },
        {
            "name": "mean first landing distance",
            "value": mean,
            "target": f"{theory:.4f} +- {3 * err:.4f}",
            "ok": abs(mean - theory) <= 3 * err,
        },
    ]


def check_energy(n=2000, dt=0.01, seed=0):
    # without drag the energy must stay the same, also after many bounces.
    # Only the interpolation at a landing can change it: it can only go down,
    # and by at most max_energy_loss(dt) at each landing
    res = run_simulation(n=n, dt=dt, p_stick=0.1, seed=seed)
    change = res["E_end"] - res["E0"]
    landings = res["bounces"] + 1
    gain = np.max(change / res["E0"])
    loss = np.max(-change / (landings * max_energy_loss(dt)))
    return [
        {
            "name": "no energy gained: max dE / E",
            "value": gain,
            "target": "<= 1e-12",
            "ok": gain <= 1e-12,
        },
        {
            "name": "energy loss / (landings g^2 dt^2 / 8)",
            "value": loss,
            "target": "<= 1",
            "ok": loss <= 1 + 1e-9,
        },
    ]


def check_symmetry(n=10000, dt=0.01, gamma=0.2, seed=0):
    # The source is the same in all directions (phi is uniform), so the final
    # positions must be symmetric around the source: mean x = 0, mean y = 0,
    # and the angle phi of the final position must be uniform.
    # We use drag here, so this also tests the runs with drag.
    res = run_simulation(n=n, dt=dt, gamma=gamma, seed=seed)
    checks = []
    for i, name in enumerate(["x", "y"]):
        values = res["final_positions"][:, i]
        err = np.std(values) / np.sqrt(n)
        checks.append(
            {
                "name": f"mean final {name}",
                "value": np.mean(values),
                "target": f"0 +- {3 * err:.4f}",
                "ok": abs(np.mean(values)) <= 3 * err,
            }
        )
    # phi must be uniform between -pi and pi. The Kolmogorov-Smirnov test
    # compares the distribution of phi with a uniform one and gives a p-value:
    # a very small p-value (< 0.01) would mean that phi is not uniform
    moved = res["final_r"] > 0
    x = res["final_positions"][moved, 0]
    y = res["final_positions"][moved, 1]
    phi = np.arctan2(y, x)
    p_value = stats.kstest(phi, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue
    checks.append(
        {
            "name": "phi is uniform: KS p-value",
            "value": p_value,
            "target": ">= 0.01",
            "ok": p_value >= 0.01,
        }
    )
    return checks


def check_random_walk(n=10000, dt=0.01, seed=0):
    # With only diffuse bounces and no drag every flight goes in a new random
    # direction, like a random walk. The steps are independent with mean 0, so
    # <r^2> = (number of flights) <R^2>, and after k bounces there are k + 1
    # flights. So r^2 / (k + 1) must have the mean <R^2> = 10 sigma^4 / g^2
    res = run_simulation(n=n, dt=dt, p_specular=0.0, seed=seed)
    values = res["final_r"] ** 2 / (res["bounces"] + 1)
    mean = np.mean(values)
    err = np.std(values) / np.sqrt(n)
    theory = mean_range_squared()
    return [
        {
            "name": "random walk: mean r^2 / (k + 1)",
            "value": mean,
            "target": f"{theory:.3f} +- {3 * err:.3f}",
            "ok": abs(mean - theory) <= 3 * err,
        }
    ]


def check_specular(n=10000, dt=0.01, seed=0):
    # With only specular bounces and no drag the particle keeps its direction
    # and every flight is the same, so it stops at r = (k + 1) R in the same
    # direction as it started
    res = run_simulation(n=n, dt=dt, p_specular=1.0, seed=seed)
    # a few very flat particles never leave the source, we skip them
    moved = res["final_r"] > 0
    expected = (res["bounces"] + 1) * range_no_drag(res["v0"])
    ratio = np.mean(res["final_r"][moved] / expected[moved])

    # the direction phi at the end must be the same as at the start
    v0 = res["v0"][moved]
    final = res["final_positions"][moved]
    phi_start = np.arctan2(v0[:, 1], v0[:, 0])
    phi_end = np.arctan2(final[:, 1], final[:, 0])
    # np.angle(np.exp(1j * d)) puts the difference d between -pi and pi
    d_phi = np.max(np.abs(np.angle(np.exp(1j * (phi_end - phi_start)))))
    return [
        {
            "name": "specular: mean r / ((k + 1) R)",
            "value": ratio,
            "target": "1 +- 1e-3",
            "ok": abs(ratio - 1) <= 1e-3,
        },
        {
            "name": "specular: max change of direction",
            "value": d_phi,
            "target": "<= 1e-9",
            "ok": d_phi <= 1e-9,
        },
    ]


def check_terminal_velocity(gamma=0.5, n=100, dt=0.1, seed=0):
    # With drag a falling particle stops speeding up when the drag force
    # gamma v is as big as gravity m g, so v_z -> -v_t = -m g / gamma.
    # We let the particles fly for the long time 40 m / gamma without the
    # floor (exp(-40) is tiny, so they must be at v_t by then)
    p = Particles.from_source(n, rng=seed, gamma=gamma)
    n_steps = int(round(40 / gamma / dt))
    for i in range(n_steps):
        p.move(dt)
    v_t = terminal_velocity(gamma=gamma)
    error = np.max(np.abs(p.v[:, 2] + v_t)) / v_t
    return [
        {
            "name": "terminal velocity: |v_z + v_t| / v_t",
            "value": error,
            "target": "<= 1e-8",
            "ok": error <= 1e-8,
        }
    ]


def run_all_checks(n=10000, dt=0.01, seed=0):
    # run all the checks, n is the number of particles for the bigger ones
    checks = []
    checks += check_range(2 * n, dt, seed)
    checks += check_energy(n // 5, dt, seed)
    checks += check_symmetry(n, dt, seed=seed)
    checks += check_random_walk(n, dt, seed)
    checks += check_specular(n, dt, seed)
    checks += check_terminal_velocity(seed=seed)
    return checks



# Plots (Kevin)


def plot_results(results, folder="figures"):
    # Kevin will write this function: the figures of one simulation
    raise NotImplementedError("the plotting functions are not ready yet")