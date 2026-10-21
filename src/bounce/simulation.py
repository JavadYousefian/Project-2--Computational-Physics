"""Main loop of the simulation and saving of the results.

The main loop follows the pseudocode of our plan::

    make the particles at the source
    while some particles are still flying:
        save the old positions and velocities
        move all flying particles one time step dt
        the particles with z < 0 landed during this step
        find where and how fast they hit the floor (linear interpolation)
        each landed particle sticks or bounces
"""

import numpy as np

from .floor import Floor
from .particles import Particles


def run_simulation(
    n=10000,
    dt=0.01,
    gamma=0.0,
    p_stick=0.2,
    p_specular=0.5,
    sigma=1.0,
    seed=None,
    method="solve_ivp",
    rtol=1e-8,
    atol=1e-10,
    max_steps=10_000_000,
    particles=None,
):
    """Run the simulation until all particles are stuck on the floor.

    Parameters
    ----------
    n : int, optional
        Number of particles.
    dt : float, optional
        Time step.
    gamma : float, optional
        Drag coefficient (0 = no air).
    p_stick : float, optional
        Probability to stick at a landing.
    p_specular : float, optional
        Probability that a bounce is specular.
    sigma : float, optional
        Width of the Maxwell-Boltzmann distribution (the temperature).
    seed : int or None, optional
        Seed of the random number generator, the same seed gives the same
        result.
    method : {"solve_ivp", "exact"}, optional
        How the particles fly, see ``Particles.move``.
    rtol : float, optional
        Relative tolerance of ``solve_ivp``.
    atol : float, optional
        Absolute tolerance of ``solve_ivp``.
    max_steps : int, optional
        Stop with an error if the loop needs more steps than this.
    particles : Particles, optional
        Start from these particles instead of the source (for the tests).

    Returns
    -------
    dict
        The results: ``final_positions`` (where each particle stopped),
        ``final_r`` (distance from the source), ``bounces`` (number of bounces
        k), ``first_landing`` and ``first_r`` (the first landing), ``v0`` and
        ``E0`` (start velocities and energies), ``E_end`` (energy at the last
        landing), ``n_steps``, ``particle_steps`` (sum of the flying particles
        over all steps), ``n_eval``, ``p_stick``, ``p_specular`` and ``gamma``.

    Raises
    ------
    ValueError
        If dt is not positive or a probability is wrong.
    RuntimeError
        If the particles are still flying after max_steps steps.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    floor = Floor(p_stick, p_specular)
    if particles is None:
        particles = Particles.from_source(n, sigma, rng, gamma)
    p = particles

    n_steps = 0
    particle_steps = 0  # sum of the flying particles over all steps (benchmark)
    while p.n_flying() > 0:
        if n_steps >= max_steps:
            raise RuntimeError("particles are still flying after max_steps")

        # save the old positions and velocities of the flying particles
        ind = p.flying()
        r_old = p.r[ind]
        v_old = p.v[ind]

        # move them one time step
        p.move(dt, method, rtol, atol)

        # the ones below the floor landed during this step
        landed = floor.find_landed(p.r[ind, 2])
        ind_land = ind[landed]
        r_land, v_land = floor.landing_point(
            r_old[landed], v_old[landed], p.r[ind_land], p.v[ind_land]
        )

        # each one sticks or bounces
        floor.land(p, ind_land, r_land, v_land, rng)

        n_steps += 1
        particle_steps += len(ind)

    final = p.r.copy()
    first = p.first_land.copy()
    return {
        "final_positions": final,  # where each particle stopped
        "final_r": np.hypot(final[:, 0], final[:, 1]),  # distance from the source
        "bounces": p.n_bounce.copy(),  # number of bounces k
        "first_landing": first,  # where it landed the first time
        "first_r": np.hypot(first[:, 0], first[:, 1]),
        "v0": p.v0.copy(),  # start velocities
        "E0": p.E0.copy(),  # start energies
        "E_end": p.energy(),  # energy at the last landing
        "n_steps": n_steps,
        "particle_steps": particle_steps,
        "n_eval": p.n_eval,
        "p_stick": p_stick,
        "p_specular": p_specular,
        "gamma": p.gamma,
    }


def save_results(results, filename):
    """Save the results in a .npz file (the numpy file format for arrays).

    Parameters
    ----------
    results : dict
        Results from ``run_simulation``.
    filename : str or pathlib.Path
        Name of the file.
    """
    np.savez(filename, **results)


def load_results(filename):
    """Load the results from a .npz file that was made by ``save_results``.

    Single numbers like n_steps come back as numpy arrays with one value, use
    int(...) to get a normal number.

    Parameters
    ----------
    filename : str or pathlib.Path
        Name of the file.

    Returns
    -------
    dict
        The results.
    """
    with np.load(filename) as data:
        return {key: data[key] for key in data.files}
