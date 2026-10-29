# Python decision-boundary examples

These are saved outputs from the local release wheel of the Python bindings
subsequently committed as `80b3773`. The environment records identify the base
commit and dirty working state at the time of execution. No algorithm code was
changed to produce these figures.

| Experiment | Training/test sizes | Seeds | Coordinate noise SD | Other parameters |
| --- | --- | --- | --- | --- |
| [Two moons](two-moons/decision-boundaries.png) | 80 / 400 | 42 / 142 | 0.13 | max_iterations=100, tolerance=0.05, nu=5; SoftBoost retries with nu=1 after AlmostSolved; SmoothBoost gamma=0.1, kappa=0.1 |
| [Noisy spirals](noisy-spirals/decision-boundaries.png) | 160 / 2,000 | 73 / 173; label flips 997 | 0.15 | max_iterations=60, tolerance=0.03, nu=16; SmoothBoost gamma=0.05, kappa=0.1 |

Parameters apply only to algorithms that expose them. TotalBoost fixes nu=1.
Both experiments use native depth-2 entropy-splitting trees. CorrectiveERLP uses
ShortStep; MLPBoost uses LineSearch. The noisy spiral experiment flips 20 training
labels, but no test labels. Test data are not used to fit or select parameters.
The dataset was made harder after inspecting an initial spiral visualization;
these examples are exploratory illustrations, not benchmark rankings.

Each directory includes `render.py`, `results.json`, and `environment.json`.
Fit timings include the Python fit call and conversion/callback overhead, and
exclude inference and plotting. Different stopping rules and equal numeric
tolerances do not imply equal optimization accuracy. The original execution
was on macOS arm64; floating-point and solver behavior can vary by platform.

To regenerate, from the repository root in a virtual environment with Rust:

```sh
python -m pip install ./python matplotlib
python img/python/two-moons/render.py
python img/python/noisy-spirals/render.py
```

The scripts overwrite their saved PNG/JSON outputs and additionally produce PDF
and NPZ files; the noisy-spiral script also generates a disagreement plot.
Inspect the recorded solver status rather than assuming every run succeeds.
