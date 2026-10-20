from bounce.analysis import plot_results
from bounce.simulation import run_simulation


def test_plot_results_makes_files(tmp_path):
    res = run_simulation(n=200, seed=1)
    files = plot_results(res, tmp_path / "figs")
    names = sorted(f.name for f in files)
    assert names == ["bounces.png", "distance_r.png", "final_positions.png"]
    for f in files:
        assert f.stat().st_size > 0