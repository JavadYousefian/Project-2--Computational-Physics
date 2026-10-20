import numpy as np
import pytest

# import functions
from bounce.sampling import maxwell_speeds, cosine_directions


def test_maxwell_speeds():
    # particle speeds must be in array of shape (N)
    rng = np.random.default_rng(0)
    v = maxwell_speeds(1000, 5.0, rng)
    assert v.shape == (1000,)
