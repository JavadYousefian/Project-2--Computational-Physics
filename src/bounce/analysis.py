# Theory formulas and validation checks (Javad). The plots at the end are
# Kevin's part.

import numpy as np

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


def bounce_probability(k, p_stick):
    # probability of exactly k bounces. Every landing is like a coin flip:
    # k times "bounce" (1 - p_stick) and then one time "stick" (p_stick)
    return (1 - p_stick) ** np.asarray(k) * p_stick


def mean_bounces(p_stick):
    # mean of P(k): the sum of k (1 - p)^k p is (1 - p) / p
    return (1 - p_stick) / p_stick


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


def run_all_checks(n=10000, dt=0.01, seed=0):
    # run all the checks, n is the number of particles for the bigger ones
    checks = []
    checks += check_range(2 * n, dt, seed)
    checks += check_energy(n // 5, dt, seed)
    return checks



# Plots (Kevin)


def plot_results(results, folder="figures"):
    # Kevin will write this function: the figures of one simulation
    raise NotImplementedError("the plotting functions are not ready yet")