# Tests for the theory formulas and the checks in analysis.py

import numpy as np
from scipy.integrate import quad

from bounce import analysis
from bounce.particles import exact_flight


def maxwell(v):
    # Maxwell-Boltzmann curve for sigma = 1 (not normalized)
    return v**2 * np.exp(-(v**2) / 2)


def maxwell_mean(power):
    # <v^power> = integral of v^power f(v) / integral of f(v)
    norm = quad(maxwell, 0, np.inf)[0]
    return quad(lambda v: v**power * maxwell(v), 0, np.inf)[0] / norm


def cosine_mean(func):
    # mean of func(theta) for the cosine law p(theta) = sin(2 theta)
    return quad(lambda t: func(t) * np.sin(2 * t), 0, np.pi / 2)[0]


def test_mean_range():
    # <R> = <v^2> <sin(2 theta)> calculated with integrals must agree with
    # the formula 3 pi sigma^2 / 4
    expected = maxwell_mean(2) * cosine_mean(lambda t: np.sin(2 * t))
    assert np.isclose(analysis.mean_range(), expected)


def test_range_formula():
    # R = 2 v_h v_z / g must be where the exact flight comes back to z = 0,
    # which happens at the time t = 2 v_z / g
    v = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 0.1]])
    ranges = analysis.range_no_drag(v)
    for i in range(2):
        r, _ = exact_flight(np.zeros((1, 3)), v[i : i + 1], 2 * v[i, 2])
        assert np.isclose(np.hypot(r[0, 0], r[0, 1]), ranges[i])


def test_bounce_probability():
    # P(k) must add up to 1, and its mean must be (1 - p) / p
    k = np.arange(2000)
    p = analysis.bounce_probability(k, 0.2)
    assert np.isclose(np.sum(p), 1)
    assert np.isclose(np.sum(k * p), analysis.mean_bounces(0.2))


def test_checks_pass():
    # the range and energy checks must pass (with fewer particles, faster)
    checks = analysis.check_range(n=2000) + analysis.check_energy(n=200)
    for check in checks:
        assert check["ok"], check