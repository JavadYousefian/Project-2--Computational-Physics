# Main loop of the simulation, like the pseudocode in our plan:
#   make the particles at the source
#   while some particles are still flying:
#       save the old positions
#       move all flying particles one time step dt
#       the particles with z < 0 landed during this step
#       find their landing points with linear interpolation
#       each landed particle sticks or bounces

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
    # n          : number of particles
    # dt         : time step
    # gamma      : drag coefficient (0 = no air)
    # p_stick    : probability to stick at a landing
    # p_specular : probability that a bounce is specular
    # sigma      : width of the Maxwell-Boltzmann distribution (temperature)
    # seed       : seed of the random number generator, same seed = same result
    # method     : "solve_ivp" or "exact", how the particles fly
    # rtol, atol : tolerances of solve_ivp
    # max_steps  : stop with an error if the loop takes too long
    # particles  : start from these particles instead of the source (for tests)
    #
    # gives back a dict with the results (see the end of the function)
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

        # save the old positions of the flying particles
        ind = p.flying()
        r_old = p.r[ind]

        # move them one time step
        p.move(dt, method, rtol, atol)

        # the ones below the floor landed during this step
        landed = floor.find_landed(p.r[ind, 2])
        ind_land = ind[landed]
        r_land = floor.landing_point(r_old[landed], p.r[ind_land])

        # each one sticks or bounces
        floor.land(p, ind_land, r_land, rng)

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