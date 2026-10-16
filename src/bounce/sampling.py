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

    Parameters
    ----------
    N : int
        number of particles
    sigma : int or float
        distribution width
    rng : numpy.random.generator
        random number generator

    Returns
    -------
    dist_arr : array of lists
        array with two-item lists containing the randomly generated velocity 
        for a given particle and its corresponding maxwell distribution value
        (in that order)
    test_lst : list
        list containing the total number of random samples generated & the number
        of samples accepted (in that order)
    """
    const = np.sqrt(2 / np.pi) * sigma**(-3)
    f = lambda v: const * (v ** 2) * np.exp(-(v ** 2) / (2 * sigma**2))
    v_max = np.sqrt(2) * (sigma)
    fv_max = f(v_max)
    # rejection sampling
    # empty array
    v_lst = []
    fv_lst = []
    # total # of runs
    runs = 0
    # samples accepted
    accs = 0
    # while there are less than N accepted particles
    while accs < N:
        runs += 1
        #random float
        v = rng.random() * (5 * sigma)
        #random float from 0 to fv_max
        fv = rng.random() * fv_max
        # if the random fv is under the f(v) curve
        if fv <= f(v):
            v_lst.append(v)
            fv_lst.append(fv)
            accs += 1
    dist_arr = np.column_stack((v_lst, fv_lst))
    test_lst = [accs, runs]
    return np.array(v_lst)

def cosine_directions(N, rng):
    """
    Obtain random theta & phi direction values for N particles
    
    Parameters
    ----------
    N : int
        number of particles
    rng : numpy.random.generator
        random number generator

    Returns
    -------
    direction_arr : array of lists
        array with two-item lists containing the orientational components, 
        theta and phi, for the particles using the relations determined
        from the cumulative distribution function

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

    # convert to cartesian (vectors for particles.py should have shape (N, 3))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    ang_dist = np.column_stack((theta, phi))
    return ang_dist


# Other to-dos for my division of labor section:
#   test the samples with the distributions
#   get the acceptance rate of the rejection sampling
