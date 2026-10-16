# bounce

Simulation of particles that are shot from a source on the floor, fly under
gravity and air drag, and stick or bounce (specular or diffuse) on the floor
`z = 0`. We use units with `m = 1` and `g = 1`.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```bash
bounce run --n 10000 --seed 1             # one simulation, saved in results.npz
bounce run --gamma 0.3 --p-specular 1.0   # with drag and only specular bounces
bounce run --plot                         # also make the figures
bounce validate                           # compare the simulation with theory
bounce validate --studies                 # also the error studies
bounce benchmark --profile                # timing for N = 1000, 10000, 100000
```

The results of `bounce run` can be loaded in Python with:

```python
from bounce.simulation import load_results
results = load_results("results.npz")
```

## Files

- `src/bounce/particles.py`: the particles and their flight (`solve_ivp`)
- `src/bounce/floor.py`: landing, sticking and bouncing
- `src/bounce/sampling.py`: random speeds and directions
- `src/bounce/simulation.py`: the main loop `run_simulation`
- `src/bounce/analysis.py`: theory, checks, error studies, benchmark, plots
- `src/bounce/cli.py`: the `bounce` command

## Tests

```bash
pytest
```

## Authors

Javad Yousefian, Alexandra VonHaven, Kevin Geumhan