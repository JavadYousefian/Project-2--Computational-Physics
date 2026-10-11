# Tests for the command line program

from bounce.cli import main
from bounce.simulation import load_results


def test_run(tmp_path):
    # "bounce run" must work and save the results in the file we ask for
    filename = tmp_path / "res.npz"
    assert main(["run", "--n", "100", "--seed", "1", "--output", str(filename)]) == 0
    assert load_results(filename)["final_positions"].shape == (100, 3)


def test_validate():
    # "bounce validate" with fewer particles must run and all checks must pass
    assert main(["validate", "--n", "1000"]) == 0