# Tests for particles.py. We check the flight with things we can calculate by hand.

import numpy as np
import pytest

from bounce.particles import Particles


def random_particles(gamma=0.0):
    # 50 particles with random positions in the air and random velocities
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 5, (50, 3))
    v = rng.normal(0, 2, (50, 3))
    return Particles(r, v, gamma=gamma)


def test_wrong_shape():
    # positions and velocities must be 3D vectors
    with pytest.raises(ValueError):
        Particles(np.zeros((3, 2)), np.zeros((3, 2)))


def test_negative_gamma():
    # negative drag is not physical
    with pytest.raises(ValueError):
        Particles(np.zeros((3, 3)), np.zeros((3, 3)), gamma=-1)


def test_projectile():
    # projectile motion without drag, start at the origin with v = (1, 0, 2).
    # z(t) = 2t - t^2/2 is zero again at t = 2 v_z / g = 4, then x = v_x t = 4
    # and v_z = -2 (same speed as at the start, but going down)
    p = Particles([[0, 0, 0]], [[1, 0, 2]])
    for i in range(400):  # 400 steps of 0.01 = time 4
        p.move(0.01)
    assert np.allclose(p.r, [[4, 0, 0]])
    assert np.allclose(p.v, [[1, 0, -2]])


def test_energy_no_drag():
    # without drag only gravity acts, so 1/2 m v^2 + m g z must stay the same
    p = random_particles()
    for i in range(200):
        p.move(0.01)
    assert np.allclose(p.energy(), p.E0, rtol=1e-12)


def test_energy_drag():
    # the drag force works against the velocity, so the energy goes down
    p = random_particles(gamma=0.5)
    E = p.energy()
    p.move(0.1)
    assert np.all(p.energy() < E)


def test_stuck():
    # stuck particles stay on the floor, only the flying ones move
    p = random_particles()
    p.stuck[::2] = True  # every second particle is stuck
    r = p.r[::2].copy()
    p.move(0.1)
    assert np.all(p.r[::2] == r)
    assert p.n_flying() == 25