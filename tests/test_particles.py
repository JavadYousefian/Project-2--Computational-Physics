# Tests for particles.py. We check the flight with things we can calculate
# by hand.

import numpy as np
import pytest

from bounce.particles import Particles, exact_flight


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


@pytest.mark.parametrize("gamma", [0.0, 0.3, 2.0])
def test_solve_ivp_vs_exact(gamma):
    # solve_ivp must give the same as the exact solution, with and without drag
    p = random_particles(gamma)
    r0 = p.r.copy()
    v0 = p.v.copy()
    for i in range(100):  # time 1
        p.move(0.01)
    r, v = exact_flight(r0, v0, 1.0, gamma=gamma)
    assert np.allclose(p.r, r, rtol=0, atol=1e-9)
    assert np.allclose(p.v, v, rtol=0, atol=1e-9)


def test_exact_method():
    # moving with method="exact" must give the same as solve_ivp
    p1 = random_particles(0.5)
    p2 = random_particles(0.5)
    for i in range(50):
        p1.move(0.02)
        p2.move(0.02, method="exact")
    assert np.allclose(p1.r, p2.r, atol=1e-10)


def test_exact_small_drag():
    # with very small drag the exact solution is almost the projectile motion
    r0 = np.zeros((1, 3))
    v0 = np.array([[1.0, 0.0, 2.0]])
    r, v = exact_flight(r0, v0, 4.0)
    assert np.allclose(r, [[4, 0, 0]])
    r2, v2 = exact_flight(r0, v0, 4.0, gamma=1e-9)
    assert np.allclose(r2, r, atol=1e-7)


def test_exact_terminal_velocity():
    # after a long fall with drag v_z = -v_t = -m g / gamma = -2 for gamma = 0.5
    r, v = exact_flight(np.zeros((1, 3)), np.zeros((1, 3)), 100.0, gamma=0.5)
    assert np.isclose(v[0, 2], -2.0)


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


def test_wrong_method():
    # only solve_ivp and exact are allowed
    with pytest.raises(ValueError):
        random_particles().move(0.1, method="euler")