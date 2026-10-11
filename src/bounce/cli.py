# Command line program. After "pip install -e ." we can type in the terminal:
#     bounce run          run one simulation and save the results
#     bounce validate     compare the simulation with the theory
#     bounce validate --studies    and also run the error studies
# (bounce benchmark will come later)

import argparse
import time

import numpy as np

from . import analysis
from .simulation import run_simulation, save_results


def make_parser():
    # all the options we can give on the command line, with their default values
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
    return parser


def do_run(args):
    # run one simulation with the options from the command line, print a short
    # summary and save everything in a .npz file
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
        # the plots are Kevin's part, until they are ready we only print a note
        try:
            analysis.plot_results(results)
        except NotImplementedError as error:
            print("no figures:", error)
    return 0


def do_validate(args):
    # run all checks and print one line for each check. We give back 1 if a
    # check failed (and 0 if all passed), so it can also be used in scripts
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


def main(argv=None):
    # this function runs when we type "bounce" in the terminal
    args = make_parser().parse_args(argv)
    if args.command == "run":
        return do_run(args)
    return do_validate(args)