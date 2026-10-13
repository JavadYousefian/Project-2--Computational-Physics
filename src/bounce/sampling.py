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
    Obtain random speed & PDF distribution values for N particles

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
    speed_dist = np.column_stack((v, speed))
    return speed_dist

def cosine_directions(N, rng):
    """
    Obtain random theta & phi direction values for N particles

    Keyword Arguments:
    N (int): number of particles
    sigma (int/float): distribution width
    rng (int/float): distribution range
    """
    # inverse CDF
    # for random floats on [0, 1)
    # particles leave soure with cosine law p(t) propto cos(t)sin(t)
    # inverse CDF gives: t = arcsin sqrt(u)
    # phi = 2piu'
    # u1 and u2: N random numbers between 0 and 1
    u1 = rng.random(N)
    u2 = rng.random(N)
    # theta direction:
    sqrtu1 = np.sqrt(u1)
    theta = np.arcsin(sqrtu1)
    phi = 2 * np.pi * u2
    ang_dist = np.column_stack((theta, phi))
    return ang_dist
