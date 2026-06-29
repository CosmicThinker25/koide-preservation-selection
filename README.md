# Supplementary material

**"Preservation versus Selection in Koide's Charged-Lepton Mass Relation"**

Author: CosmicThinker25 (independent researcher).
Technical and computational assistance: Claude (Anthropic).

## Scripts (run in this order)

1. `koide_sweep.py` — S3-invariant gradient-flow sweep on S^2_+; geometric null; per-trajectory and per-potential Koide-band frequencies across four windows. Writes `sweep_data.npz`. **The full run takes several minutes; use `--fast` for a smoke test.**

2. `koide_running.py` — one-loop SM running of K(mu) from MS-bar lepton masses at M_Z (Sec. 5 table).

3. `make_figures.py` — reads `sweep_data.npz` and writes the three figures (`fig1_sphere.pdf`, `fig2_hist.pdf`, `fig3_null.pdf`). Requires matplotlib.

```bash
python koide_sweep.py            # full run -> paper tables (Sec. 8) + sweep_data.npz
python koide_sweep.py --fast     # quick smoke test
python koide_running.py          # running table (Sec. 5)
python make_figures.py           # figures (needs sweep_data.npz from step 1)
```

## Reproducibility

Seeds are fixed and explicit (no `hash()`); sweep results are byte-identical across runs and machines, independent of `PYTHONHASHSEED`.

Fixed seeds: null=20240628, gaussian=20240728, uniform=20240829.

## Environment

Tested with: Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8.

NumPy is used for the sweep, SciPy for the running calculation, and Matplotlib for the figures.

## Scope

Negative, bounded methodological note. The sweep is exploratory within a low-order polynomial ansatz in the invariants `(e2,e3)`; it is not an impossibility theorem.

The boundary is handled by reflection `(x_i -> |x_i|)`, which biases the vertex rate but not the relative under-population of the Koide band. An internal-coordinate integrator is left for future work.
