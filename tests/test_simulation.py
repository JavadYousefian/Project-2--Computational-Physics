import numpy as np
import pytest

from bounce.simulation import load_results, run_simulation, save_results


def range_formula(v):
    """Range without drag: R = v^2 sin(2 theta) / g = 2 v_h v_z / g, with g = 1."""
    return 2 * np.hypot(v[:, 0], v[:, 1]) * v[:, 2]


def test_all_stuck_on_floor():
    """At the end every particle is stuck and lies on the floor (z = 0)"""
    res = run_simulation(n=300, seed=1)
    assert res["final_positions"].shape == (300, 3)
    assert np.all(res["final_positions"][:, 2] == 0)
    assert not np.any(np.isnan(res["first_landing"]))


def test_same_seed_same_result():
    """Same seed -> same random numbers -> exactly the same result."""
    res1 = run_simulation(n=200, seed=5)
    res2 = run_simulation(n=200, seed=5)
    assert np.all(res1["final_positions"] == res2["final_positions"])
    assert np.all(res1["bounces"] == res2["bounces"])


def test_p_stick_one():
    """With p_stick = 1 every particle sticks at its first landing."""
    res = run_simulation(n=500, p_stick=1.0, seed=2)
    assert np.all(res["bounces"] == 0)
    assert np.all(res["final_positions"] == res["first_landing"])


def test_first_landing():
    """The first landing must agree with R = v^2 sin(2 theta) / g.

    We only see the landing at the end of a step, so the error can not be more than the
    horizontal distance of one step, v_h dt.
    """
    dt = 0.005
    res = run_simulation(n=1000, dt=dt, p_stick=1.0, seed=3)
    error = np.abs(res["first_r"] - range_formula(res["v0"]))
    v_h = np.hypot(res["v0"][:, 0], res["v0"][:, 1])
    assert np.all(error <= v_h * dt)
    assert np.mean(error) < 1e-4


def test_specular_goes_straight():
    """With only specular bounces and no drag all flights of a particle are the same, so
    it stops at r = (k + 1) R.

    We skip the very short flights.
    """
    res = run_simulation(n=500, p_stick=0.3, p_specular=1.0, seed=4)
    expected = (res["bounces"] + 1) * range_formula(res["v0"])
    far = res["final_r"] > 0.1
    assert np.allclose(res["final_r"][far], expected[far], rtol=1e-2)


def test_energy_not_increasing():
    """Without drag the energy can not go up.

    It can go down a tiny bit at the landings because of the interpolation (see
    analysis.max_energy_loss)
    """
    res = run_simulation(n=300, seed=6)
    assert np.all(res["E_end"] <= res["E0"] + 1e-12)


def test_drag_makes_shorter():
    """Air drag slows the particles down, so they land closer to the source."""
    free = run_simulation(n=500, p_stick=1.0, seed=7)
    drag = run_simulation(n=500, p_stick=1.0, gamma=0.5, seed=7)
    assert np.mean(drag["final_r"]) < np.mean(free["final_r"])


def test_exact_method_same():
    """Solve_ivp and the exact solution must give the same simulation."""
    res1 = run_simulation(n=200, gamma=0.3, seed=8)
    res2 = run_simulation(n=200, gamma=0.3, seed=8, method="exact")
    assert np.all(res1["bounces"] == res2["bounces"])
    assert np.allclose(res1["final_positions"], res2["final_positions"], atol=1e-8)


def test_wrong_input():
    """P_stick = 0 never stops, dt must be positive, and max_steps stops the loop."""
    with pytest.raises(ValueError):
        run_simulation(n=10, p_stick=0)
    with pytest.raises(ValueError):
        run_simulation(n=10, dt=0)
    with pytest.raises(RuntimeError):
        run_simulation(n=10, max_steps=3, seed=0)


def test_save_and_load(tmp_path):
    """Saving and loading must give back the same results.

    tmp_path is a temporary folder that pytest makes for the test.
    """
    res = run_simulation(n=100, seed=9)
    filename = tmp_path / "res.npz"
    save_results(res, filename)
    loaded = load_results(filename)
    assert np.all(loaded["final_positions"] == res["final_positions"])
    assert int(loaded["n_steps"]) == res["n_steps"]
