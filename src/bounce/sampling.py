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
    f = lambda v: (v ** 2) * np.exp(-(v ** 2) / (2 * sigma**2))
    v_max = np.sqrt(2) * (sigma)
    fv_max = f(v_max)
    # rejection sampling
    # empty array
    v_lst = []
    fv_lst = []
    counts = 0
    while counts < N:
        #random float from 0 to 1
        v = random.random()
        #random float from 0 to fv_max
        fv = random.random() * fv_max
        # if the random fv is under the f(v) curve
        if fv <= f(v):
            v_lst.append(v)
            fv_lst.append(fv)
            counts += 1
    dist_arr = np.column_stack(v_lst, fv_lst)
    # N random velocities between 0 & v_max
    # v = rng.random(N) * v_max
    # # speed according to the maxwell distribution
    # fv = rng.random(N) * fv_max # numpy array    # put v and speed into a state vector
    # dist_arr = np.column_stack((v, fv))
    # mask = np.array([j <= f(i) for i, j in dist_arr])
    # return dist_arr[mask]

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
