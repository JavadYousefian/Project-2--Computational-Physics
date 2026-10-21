from bounce import analysis
from bounce.analysis import plot_results
from bounce.simulation import run_simulation


def test_plot_results_makes_files(tmp_path):
    """plot_results must save the 3 figures."""
    res = run_simulation(n=200, seed=1)
    files = plot_results(res, tmp_path / "figs")
    names = sorted(f.name for f in files)
    assert names == ["bounces.png", "distance_r.png", "final_positions.png"]
    for f in files:
        assert f.stat().st_size > 0


def test_check_bounces_pass():
    """The number of bounces must follow P(k), for two values of p_stick."""
    for p_stick in (0.2, 0.5):
        for check in analysis.check_bounces(n=2000, p_stick=p_stick):
            assert check["ok"], check


def test_statistical_error_slope(tmp_path):
    """The statistical error must go like 1 / sqrt(N)."""
    rows, slope = analysis.statistical_error_study(sizes=(300, 1000, 3000))
    assert abs(slope + 0.5) < 0.2
    filename = analysis.plot_statistical_error(rows, tmp_path)
    assert filename.stat().st_size > 0
