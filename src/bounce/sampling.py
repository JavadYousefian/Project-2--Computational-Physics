import numpy as np
import random

# maxwell_speeds-- name Javad assigned for speeds
# cosine_directions -- inverse CDF
# I'M MAKING THE FIRST STEP! I generate the random numbers and get the positions
# python -m pip install -e ".[dev]"
# python -m pytest
# src/bounce/sampling.py

# np.array([r1, v1], [r2, v2] ….. [rn, vn])

def maxwell_speeds(N, sigma, rng):
    """
    Obtain vectors with random 1D position & velocity values

    Keyword Arguments:
    N (int): number of particles
    sigma (int/float): distribution width
    rng (int/float): distribution range
    """
    # rejection sampling
    # N random velocities between 0 & 1
    v = rng.random(N)   # numpy array
    # speed according to the maxwell distribution
    speed = (v ** 2) * np.exp(-(v ** 2) / (2 * sigma**2))
    # put v and speed into a state vector
    state_vecs = np.column_stack((v, speed))
    return state_vecs

def cosine_directions(N, range):
    # inverse CDF
    # for random floats on [0, 1)
    # particles leave soure with cosine law p(t) propto cos(t)sin(t)
    # inverse CDF gives: t = arcsin sqrt(u)
    # phi = 2piu'
    pass