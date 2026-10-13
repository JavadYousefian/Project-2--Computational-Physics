import numpy as np
import random

# maxwell_speeds-- name Javad assigned for speeds
# cosine_directions -- inverse CDF
# I'M MAKING THE FIRST STEP! I generate the random numbers and get the positions
# python -m pip install -e ".[dev]"
# python -m pytest
# src/bounce/sampling.py

# np.array([r1, v1], [r2, v2] ….. [rn, vn])

def maxwell_speeds(N, dt, gamma, p_s, p_b, v):
    """
    Obtain vectors with random 1D position & velocity values

    Keyword Arguments:
    N (int): number of particles
    dt (int/float): time interval
    gamma (int/float): drag coefficient
    p_s (int/float): probability the particle will stick
    p_s (int/float): probability the particle will bounce
    v (int/float): particle speed
    """
    # rejection sampling

    # get the r & v values
    for n in range(np):
        # random number between 0 and 1
        r = random.random()

def cosine_directions():
    # inverse CDF
    # for random floats on [0, 1)
    # particles leave soure with cosine law p(t) propto cos(t)sin(t)
    # inverse CDF gives: t = arcsin sqrt(u)
    # phi = 2piu'
    pass