# Tests for floor.py

import numpy as np
import pytest

from bounce.floor import Floor
from bounce.particles import Particles


def landed_particles(n=200):
    # n particles on the floor that move down (v_z < 0), like just after landing
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, (n, 3))
    v[:, 2] = -np.abs(v[:, 2])
    return Particles(np.zeros((n, 3)), v), rng


def test_wrong_p_stick():
    # p_stick = 0 would mean the particles never stop
    with pytest.raises(ValueError):
        Floor(0)
    with pytest.raises(ValueError):
        Floor(1.5)


def test_find_landed():
    # only particles with z < 0 landed (z = 0 is still on the floor)
    floor = Floor(0.5)
    landed = floor.find_landed(np.array([1.0, 0.0, -1e-12, -3.0]))
    assert list(landed) == [False, False, True, True]


def test_landing_point():
    # z goes from 1 to -2 in the step, so z = 0 at 1/3 of the step
    floor = Floor(0.5)
    r_old = np.array([[0.0, 0.0, 1.0]])
    r_new = np.array([[3.0, 1.0, -2.0]])
    r = floor.landing_point(r_old, r_new)
    assert np.allclose(r, [[1, 1 / 3, 0]])


def test_all_stick():
    # with p_stick = 1 every particle sticks at the first landing
    p, rng = landed_particles()
    ind = np.arange(len(p))
    Floor(1.0).land(p, ind, p.r.copy(), rng)
    assert np.all(p.stuck)
    assert np.all(p.n_bounce == 0)
    assert np.all(p.first_land == p.r)


def test_specular():
    # specular bounce: v_z changes sign, v_x and v_y stay the same
    p, rng = landed_particles()
    v_in = p.v.copy()
    ind = np.arange(len(p))
    Floor(1e-12).land(p, ind, p.r.copy(), rng)  # p_stick almost 0, so all bounce
    assert np.allclose(p.v[:, :2], v_in[:, :2])
    assert np.allclose(p.v[:, 2], -v_in[:, 2])
    assert np.all(p.n_bounce == 1)


def test_first_landing_saved_once():
    # the first landing must not change at the next landings
    p, rng = landed_particles(10)
    floor = Floor(1e-12)
    ind = np.arange(10)
    floor.land(p, ind, np.ones((10, 3)), rng)
    floor.land(p, ind, 2 * np.ones((10, 3)), rng)
    assert np.all(p.first_land == 1)
    assert np.all(p.n_bounce == 2)


def test_stick_probability():
    # with many particles the fraction that sticks must be close to p_stick
    p, rng = landed_particles(20000)
    Floor(0.3).land(p, np.arange(len(p)), p.r.copy(), rng)
    assert abs(np.mean(p.stuck) - 0.3) < 0.02