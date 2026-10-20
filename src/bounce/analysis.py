# Theory formulas, validation checks, error studies and benchmark (Javad).
# The plots at the end are Kevin's part.

import cProfile
import io
import pstats
import time

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from .particles import Particles, exact_flight
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


def mean_energy_loss(dt, g=1.0, m=1.0):
    # The height of the particle at the interpolated time is g s (dt - s) / 2,
    # where s is the time in the step when it lands. s can be anywhere in the
    # step with the same chance, and the mean of s (dt - s) is dt^2 / 6, so the
    # mean energy loss at one landing is m g^2 dt^2 / 12
    return m * g**2 * dt**2 / 12


def mean_range_error(dt, g=1.0):
    # The interpolated landing is too early by (height / v_z), so the particle
    # lands too short by v_h * height / v_z = tan(theta) g s (dt - s) / 2.
    # The mean over s gives tan(theta) g dt^2 / 12, and for the cosine law the
    # mean of tan(theta) is pi / 2, so the mean error is pi g dt^2 / 24
    return np.pi * g * dt**2 / 24



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



# Error studies



def roundoff_study(steps_list=(10, 100, 1000, 10000), t_total=2.0, n=1000, seed=0):
    # Finite precision: the computer keeps only about 16 digits of a number, so
    # every calculation has a tiny round-off error. Without drag RK45 is exact
    # for the parabola, so if the energy changes in a flight without the floor
    # it can only come from round-off. We fly the same time t_total with more
    # and more steps and look how the energy error grows.
    rows = []
    for steps in steps_list:
        p = Particles.from_source(n, rng=seed)
        dt = t_total / steps
        for i in range(steps):
            p.move(dt)
        change = np.abs(p.energy() - p.E0) / p.E0
        rows.append(
            {
                "steps": steps,
                "dt": dt,
                "max dE/E": np.max(change),
                "mean dE/E": np.mean(change),
            }
        )
    return rows


def integration_study(
    rtols=(1e-3, 1e-4, 1e-6, 1e-8, 1e-10), gamma=0.5, t_total=5.0, dt=0.01, seed=0
):
    # Error of solve_ivp with drag, compared with the exact solution.
    # "1 call": one solve_ivp call for the whole time, so the solver chooses
    #           its own steps from rtol and the error should follow rtol
    # "steps":  one call per dt like in our simulation. dt is so small that
    #           the error is tiny for every rtol
    # (we use atol = rtol / 1000)
    start = Particles.from_source(1000, rng=seed, gamma=gamma)
    r_exact, _ = exact_flight(start.r, start.v, t_total, gamma=gamma)
    y0 = np.concatenate((start.r.ravel(), start.v.ravel()))
    rows = []
    for rtol in rtols:
        atol = rtol / 1000
        sol = solve_ivp(start.rhs, (0, t_total), y0, rtol=rtol, atol=atol)
        r_one = sol.y[: len(y0) // 2, -1].reshape(-1, 3)

        stepped = Particles(start.r, start.v, gamma=gamma)
        for i in range(int(round(t_total / dt))):
            stepped.move(dt, rtol=rtol, atol=atol)
        rows.append(
            {
                "rtol": rtol,
                "error (1 call)": np.max(np.abs(r_one - r_exact)),
                "nfev (1 call)": sol.nfev,
                "error (steps)": np.max(np.abs(stepped.r - r_exact)),
                "nfev (steps)": stepped.n_eval,
            }
        )
    return rows


def timestep_study(dts=(0.04, 0.02, 0.01, 0.005, 0.0025), n=10000, seed=0):
    # Error from the time step: we only find the landing at the end of a step,
    # with linear interpolation. We use p_stick = 1 and no drag, so every
    # particle lands once and the exact answer is R = v^2 sin(2 theta) / g.
    # Both errors should go like dt^2 (order 2), see mean_range_error and
    # mean_energy_loss for the theory
    rows = []
    for dt in dts:
        res = run_simulation(n=n, dt=dt, p_stick=1.0, seed=seed)
        error = np.abs(res["first_r"] - range_no_drag(res["v0"]))
        loss = res["E0"] - res["E_end"]
        rows.append(
            {
                "dt": dt,
                "mean error": np.mean(error),
                "theory": mean_range_error(dt),
                "max error": np.max(error),
                "mean dE": np.mean(loss),
                "theory dE": mean_energy_loss(dt),
            }
        )
    # fit a line to log(error) against log(dt), the slope is the order
    log_dt = np.log(dts)
    order_range = np.polyfit(log_dt, np.log([r["mean error"] for r in rows]), 1)[0]
    order_energy = np.polyfit(log_dt, np.log([r["mean dE"] for r in rows]), 1)[0]
    return rows, order_range, order_energy


def print_table(rows):
    # print a list of dicts as a simple table
    keys = list(rows[0].keys())
    width = max(12, max(len(key) for key in keys) + 1)
    print("".join(f"{key:>{width}}" for key in keys))
    for row in rows:
        line = ""
        for key in keys:
            value = row[key]
            if isinstance(value, (int, np.integer)):
                line += f"{value:>{width}d}"
            else:
                line += f"{value:>{width}.3g}"
        print(line)


# Benchmark and profiling



def benchmark_step(sizes=(1000, 10000, 100000), dt=0.01, repeats=20, seed=0):
    # Time of one move() when all n particles are flying. The particles do not
    # interact, so the work should grow like n (linear). We repeat it and keep
    # the best time, because other programs on the computer can slow it down
    rows = []
    for n in sizes:
        p = Particles.from_source(n, rng=seed)
        times = []
        for i in range(repeats):
            t0 = time.perf_counter()
            p.move(dt)
            times.append(time.perf_counter() - t0)
        best = min(times)
        rows.append({"n": n, "time/step": best, "time/particle": best / n})
    return rows


def benchmark_run(sizes=(1000, 10000, 100000), seed=0):
    # Time of a full simulation with the default settings. The number of steps
    # is set by the slowest particles (the last ones that stick), and in most
    # of these steps only a few particles are still flying
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        res = run_simulation(n=n, seed=seed)
        total = time.perf_counter() - t0
        rows.append(
            {
                "n": n,
                "time": total,
                "steps": res["n_steps"],
                "particle steps": res["particle_steps"],
                "time/particle step": total / res["particle_steps"],
            }
        )
    return rows


def scaling_exponent(rows, key):
    # fit time ~ n^a, so log(time) = a log(n) + b, and give back a
    n = [row["n"] for row in rows]
    t = [row[key] for row in rows]
    return np.polyfit(np.log(n), np.log(t), 1)[0]


def profile_run(n=10000, top=15, seed=0):
    # run one simulation with cProfile, which measures how much time is spent
    # in every function, and give back the report of the slowest ones
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(n=n, seed=seed)
    profiler.disable()
    output = io.StringIO()
    report = pstats.Stats(profiler, stream=output)
    report.sort_stats("cumulative").print_stats(top)
    return output.getvalue()



# Plots (Kevin)
# Written by kevin Geumhan

from pathlib import Path
from matplotlib.figure import Figure

def title_text(results):
    #literally the title for the figures
    n = len(results['final_r'])
    return(
        f"N = {n}, p_stick = {float(results['p_stick']):g}, "
        f"p_spec = {float(results['p_specular']):g}, "
        f"gamma = {float(results['gamma']):g}"
    )

def plot_final_positions(results, folder):

    pos = results['final_positions']
    n = len(pos)
    fig = Figure(figsize=(5.5,5))
    ax = fig.add_subplot()

    alpha = min(1.0, 3000 / n)
    ax.scatter(pos[:, 0], pos[:,1], s=2, alpha=alpha, lw=0)
    ax.plot(0, 0, 'r+', ms=12, label='source')
    lim = np.percentile(results["final_r"], 99)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title_text(results), fontsize=9)
    ax.legend(loc="upper right")
    filename = Path(folder) / "final_positions.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    return filename

def plot_distance(results, folder):

    final_r = results["final_r"]
    first_r = results["first_r"]
    bins = np.linspace(0, np.percentile(final_r, 99), 60)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot()
    ax.hist(final_r, bins=bins, density=True, alpha=0.6, label="final position")
    ax.hist(
        first_r,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label="first landing",
    )
    mean_final = np.mean(final_r)
    mean_first = np.mean(first_r)
    ax.axvline(mean_final, color="C0", ls="--", label=f"mean final = {mean_final:.3f}")
    ax.axvline(mean_first, color="C1", ls="--", label=f"mean first = {mean_first:.3f}")
    if float(results["gamma"]) == 0:
        theory = mean_range()
        ax.axvline(theory, color="k", ls=":", label=f"theory <R> = {theory:.3f}")
    ax.set_xlabel("distance r from the source")
    ax.set_ylabel("probability density")
    ax.set_title(title_text(results), fontsize=9)
    ax.legend(fontsize=8)
    filename = Path(folder) / "distance_r.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    return filename


def plot_bounces(results, folder):
    
    k = results["bounces"]
    n = len(k)
    p_stick = float(results["p_stick"])
    counts = np.bincount(k)
    k_values = np.arange(len(counts))
    seen = counts > 0 
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot()
    ax.errorbar(
        k_values[seen],
        counts[seen] / n,
        yerr=np.sqrt(counts[seen]) / n,
        fmt="o",
        ms=4,
        capsize=2,
        label="simulation",
    )
    theory = bounce_probability(k_values, p_stick)
    ax.plot(k_values, theory, "k-", label="theory $(1 - p)^k p$")
    ax.set_yscale("log")
    ax.set_xlabel("number of bounces k")
    ax.set_ylabel("P(k)")
    ax.set_title(
        title_text(results)
        + f"\nmean k = {np.mean(k):.3f}, theory = {mean_bounces(p_stick):.3f}",
        fontsize=9,
    )
    ax.legend()
    filename = Path(folder) / "bounces.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    return filename


def plot_results(results, folder="figures"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = [plot_final_positions(results, folder)]
    files.append(plot_distance(results, folder))
    files.append(plot_bounces(results, folder))
    print(f'figures saved in {folder}')
    return files

def check_bounces(n=10000, dt=0.01, p_stick=0.2, seed=0):

    res = run_simulation(n=n, dt=dt, p_stick=p_stick, seed=seed)
    k = res["bounces"]
    mean = np.mean(k)
    err = np.std(k) / np.sqrt(n)
    theory = mean_bounces(p_stick)

    # the largest k_max where n P(k_max - 1) is still >= 5
    k_max = 1
    while n * bounce_probability(k_max, p_stick) >= 5:
        k_max += 1
    observed = np.bincount(np.minimum(k, k_max), minlength=k_max + 1)
    expected = n * bounce_probability(np.arange(k_max), p_stick)
    # P(k >= k_max) = (1 - p_stick)^k_max (sum of the geometric series)
    expected = np.append(expected, n * (1 - p_stick) ** k_max)
    p_value = stats.chisquare(observed, expected).pvalue
    return [
        {
            "name": "mean number of bounces",
            "value": mean,
            "target": f"{theory:.4f} +- {3 * err:.4f}",
            "ok": abs(mean - theory) <= 3 * err,
        },
        {
            "name": "bounces: chi-square p-value of P(k)",
            "value": p_value,
            "target": ">= 0.003",
            # 0.003 is the same chance to fail by bad luck as 3 sigma
            "ok": p_value >= 0.003,
        },
    ]


def statistical_error_study(sizes=(1000, 10000, 100000), dt=0.01, seed=0):

    rows = []
    for n in sizes:
        res = run_simulation(n=n, dt=dt, seed=seed)
        rows.append(
            {
                "n": n,
                "mean r": np.mean(res["final_r"]),
                "error r": np.std(res["final_r"]) / np.sqrt(n),
                "mean R": np.mean(res["first_r"]),
                "error R": np.std(res["first_r"]) / np.sqrt(n),
                "|R - theory|": abs(np.mean(res["first_r"]) - mean_range()),
            }
        )
    # the slope of log(error) against log(n), like scaling_exponent for the time
    return rows, scaling_exponent(rows, "error r")


def plot_statistical_error(rows, folder="figures"):

    n = np.array([row["n"] for row in rows])
    error_r = np.array([row["error r"] for row in rows])
    error_first = np.array([row["error R"] for row in rows])
    deviation = np.array([row["|R - theory|"] for row in rows])
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot()
    ax.loglog(n, error_r, "o-", label="std / sqrt(N), final r")
    ax.loglog(n, error_first, "s-", label="std / sqrt(N), first landing R")
    ax.loglog(n, deviation, "^", ms=8, label="|mean R - 3 pi / 4| (real error)")
    ax.loglog(n, error_r[0] * np.sqrt(n[0] / n), "k--", label="slope -1/2")
    ax.set_xlabel("number of particles N")
    ax.set_ylabel("error of the mean distance")
    ax.legend(fontsize=8)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / "statistical_error.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    return filename

#testing testing