"""Command line program ``bounce``.

After ``pip install -e .`` we can type in the terminal::

    bounce run                   run one simulation and save the results
    bounce run --plot            and also make the figures
    bounce validate              compare the simulation with the theory
    bounce validate --studies    and also run the error studies
    bounce benchmark             measure the time for N = 1000, 10000, 100000
    bounce benchmark --profile   and also run cProfile
"""

import argparse
import time

import numpy as np

from . import analysis
from .simulation import run_simulation, save_results


def make_parser():
    """Make the parser for the command line options.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the commands run, validate and benchmark and their default
        values.
    """
    parser = argparse.ArgumentParser(
        prog="bounce", description="Particles bouncing and sticking on a floor."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one simulation")
    run.add_argument("--n", type=int, default=10000, help="number of particles")
    run.add_argument("--dt", type=float, default=0.01, help="time step")
    run.add_argument("--gamma", type=float, default=0.0, help="drag coefficient")
    run.add_argument("--p-stick", type=float, default=0.2)
    run.add_argument("--p-specular", type=float, default=0.5)
    run.add_argument("--sigma", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--method", default="solve_ivp", choices=["solve_ivp", "exact"])
    run.add_argument("--output", default="results.npz", help="output file")
    run.add_argument("--plot", action="store_true", help="make the figures")

    validate = commands.add_parser("validate", help="compare with the theory")
    validate.add_argument("--n", type=int, default=10000, help="number of particles")
    validate.add_argument("--dt", type=float, default=0.01, help="time step")
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--studies", action="store_true", help="error studies")

    benchmark = commands.add_parser("benchmark", help="measure the time")
    benchmark.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 10000, 100000]
    )
    benchmark.add_argument("--profile", action="store_true", help="run cProfile")
    benchmark.add_argument("--profile-n", type=int, default=10000)
    return parser


def do_run(args):
    """Run one simulation, print a short summary and save the results.

    With --plot it also makes the figures.

    Parameters
    ----------
    args : argparse.Namespace
        Options from the command line.

    Returns
    -------
    int
        Exit code, 0 if everything worked.
    """
    t0 = time.perf_counter()
    results = run_simulation(
        n=args.n,
        dt=args.dt,
        gamma=args.gamma,
        p_stick=args.p_stick,
        p_specular=args.p_specular,
        sigma=args.sigma,
        seed=args.seed,
        method=args.method,
    )
    t = time.perf_counter() - t0
    save_results(results, args.output)
    print(f"{args.n} particles, {results['n_steps']} steps, {t:.2f} s")
    print(f"mean number of bounces: {np.mean(results['bounces']):.3f}")
    print(f"mean final distance:    {np.mean(results['final_r']):.3f}")
    print(f"results saved in {args.output}")
    if args.plot:
        # make the figures of this run
        try:
            analysis.plot_results(results)
        except NotImplementedError as error:
            print("no figures:", error)
    return 0


def do_validate(args):
    """Run all validation checks and print one line for every check.

    With --studies it also runs the error studies. The exit code is 1 if a
    check failed, so it can also be used in scripts.

    Parameters
    ----------
    args : argparse.Namespace
        Options from the command line.

    Returns
    -------
    int
        Exit code, 0 if everything worked.
    """
    checks = analysis.run_all_checks(args.n, args.dt, args.seed)
    n_failed = 0
    for check in checks:
        if check["ok"]:
            status = "PASS"
        else:
            status = "FAIL"
            n_failed += 1
        print(
            f"[{status}] {check['name']:<40} {check['value']:<12.6g}"
            f" target {check['target']}"
        )
    print(f"{len(checks) - n_failed} of {len(checks)} checks passed")

    if args.studies:
        print("\nRound-off: flight without drag and without the floor")
        analysis.print_table(analysis.roundoff_study(seed=args.seed))
        print("\nsolve_ivp with drag compared with the exact solution")
        analysis.print_table(analysis.integration_study(seed=args.seed))
        print("\nTime step: first landing and energy loss")
        rows, order_range, order_energy = analysis.timestep_study(seed=args.seed)
        analysis.print_table(rows)
        print(f"order: range {order_range:.2f}, energy {order_energy:.2f}")

    if n_failed > 0:
        return 1
    return 0


def do_benchmark(args):
    """Measure the time for different numbers of particles.

    With --profile it also runs cProfile to find the slowest functions.

    Parameters
    ----------
    args : argparse.Namespace
        Options from the command line.

    Returns
    -------
    int
        Exit code, 0 if everything worked.
    """
    print("One step with all particles flying")
    rows = analysis.benchmark_step(args.sizes)
    analysis.print_table(rows)
    print(f"time/step ~ N^{analysis.scaling_exponent(rows, 'time/step'):.2f}")

    print("\nFull simulation")
    rows = analysis.benchmark_run(args.sizes)
    analysis.print_table(rows)
    print(f"time ~ N^{analysis.scaling_exponent(rows, 'time'):.2f}")

    if args.profile:
        print(f"\ncProfile with N = {args.profile_n}")
        print(analysis.profile_run(args.profile_n))
    return 0


def main(argv=None):
    """Start the program, this runs when we type ``bounce`` in the terminal.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, the real command line is used.

    Returns
    -------
    int
        Exit code, 0 if everything worked.
    """
    args = make_parser().parse_args(argv)
    if args.command == "run":
        return do_run(args)
    if args.command == "validate":
        return do_validate(args)
    return do_benchmark(args)
