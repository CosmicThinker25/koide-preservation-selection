#!/usr/bin/env python3
"""
Reproducible sweep for "Preservation versus Selection in Koide's relation".
S3-invariant gradient flows V(e2,e3) on the positive sphere S^2_+.

Deterministic, fixed seeds (no hash()). Reports per-trajectory AND
per-potential band frequencies across several windows, plus the geometric
null under the same sampling measure.

Usage:
    python koide_sweep.py            # full run (paper numbers)
    python koide_sweep.py --fast     # quick smoke test

Tested with NumPy >= 1.24.

NOTE: the full run may take several minutes; use --fast for a smoke test.
"""
import sys
import numpy as np

SEED = 20240627
SEEDS = {                      # explicit, machine-independent seeds
    "null":          SEED + 1,
    "gaussian":      SEED + 101,
    "uniform[-1,1]": SEED + 202,
}
WINDOWS = [(0.22, 0.28), (0.23, 0.27), (0.24, 0.26), (0.245, 0.255)]

FAST = "--fast" in sys.argv
M_NULL      = 200_000 if FAST else 4_000_000
N_POT_GAUSS =     200 if FAST else 1500
N_POT_UNIF  =     200 if FAST else 1200
N_INI       = 8
DT, STEPS   = 0.02, (400 if FAST else 1500)


def e2_of(x):
    return x[..., 0]*x[..., 1] + x[..., 0]*x[..., 2] + x[..., 1]*x[..., 2]


def geometric_null(rng, M):
    x = np.abs(rng.normal(size=(M, 3)))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    e2 = e2_of(x)
    return {w: float(np.mean((e2 > w[0]) & (e2 < w[1]))) for w in WINDOWS}, float(e2.mean())


def flow(coef, x0, dt=DT, steps=STEPS):
    a, b, c, d, f = coef
    x = x0.copy()
    for _ in range(steps):
        x = np.abs(x); x /= np.linalg.norm(x, axis=1, keepdims=True)
        E2 = e2_of(x); E3 = x[:, 0]*x[:, 1]*x[:, 2]
        de2 = np.stack([x[:, 1]+x[:, 2], x[:, 0]+x[:, 2], x[:, 0]+x[:, 1]], 1)
        de3 = np.stack([x[:, 1]*x[:, 2], x[:, 0]*x[:, 2], x[:, 0]*x[:, 1]], 1)
        g = ((a + 2*b*E2 + f*E3)[:, None])*de2 + ((c + 2*d*E3 + f*E2)[:, None])*de3
        g = g - (np.sum(g*x, 1, keepdims=True))*x
        x = x - dt*g
    x = np.abs(x); x /= np.linalg.norm(x, axis=1, keepdims=True)
    return e2_of(x)                       # final e2 for the N_ini trajectories


def sweep(sampler, n_pot, rng):
    """Return an (n_pot x N_ini) matrix of final e2 values (keeps per-potential structure)."""
    rows = []
    for i in range(n_pot):
        if i and i % 300 == 0:
            print(f"   ... {i}/{n_pot}", flush=True)
        coef = sampler(rng)
        x0 = np.abs(rng.normal(size=(N_INI, 3)))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        rows.append(flow(coef, x0))
    return np.array(rows)                  # shape (n_pot, N_ini)


def band_freqs(e2_matrix, w):
    in_band = (e2_matrix > w[0]) & (e2_matrix < w[1])
    return float(np.mean(in_band)), float(np.mean(np.any(in_band, axis=1)))


def classify(e2_flat, w):
    demo = np.mean(e2_flat > 0.95)
    vert = np.mean(e2_flat < 0.05)
    koide = np.mean((e2_flat > w[0]) & (e2_flat < w[1]))
    return demo, vert, koide, 1 - demo - vert - koide


if __name__ == "__main__":
    print(f"mode={'FAST' if FAST else 'FULL'}  seeds={SEEDS}")
    _rng_null = np.random.default_rng(SEEDS["null"])
    _xn = np.abs(_rng_null.normal(size=(M_NULL,3))); _xn /= np.linalg.norm(_xn,axis=1,keepdims=True)
    _e2_null = e2_of(_xn)
    null = {w: float(np.mean((_e2_null>w[0])&(_e2_null<w[1]))) for w in WINDOWS}
    e2bar = float(_e2_null.mean())
    print(f"\nGEOMETRIC NULL  (<e2>={e2bar:.4f}):")
    for w in WINDOWS:
        print(f"   P({w[0]}<e2<{w[1]}) = {100*null[w]:.3f}%")

    configs = [("gaussian",      lambda r: r.normal(size=5),     N_POT_GAUSS),
               ("uniform[-1,1]", lambda r: r.uniform(-1, 1, 5),  N_POT_UNIF)]
    saved = {}
    for name, sampler, n_pot in configs:
        rng = np.random.default_rng(SEEDS[name])
        e2m = sweep(sampler, n_pot, rng)
        saved[name] = e2m
        d, v, k, o = classify(e2m.ravel(), (0.22, 0.28))
        print(f"\n[{name}]  {n_pot} potentials x {N_INI} = {e2m.size} trajectories")
        print(f"   democratic={100*d:.2f}%  vertices={100*v:.2f}%  other={100*o:.2f}%")
        print("   Koide band (per-traj | per-pot | null | traj/null):")
        for w in WINDOWS:
            tf, pf = band_freqs(e2m, w)
            print(f"     {str(w):16s}: {100*tf:5.2f}% | {100*pf:5.2f}% | "
                  f"{100*null[w]:5.2f}% | {tf/null[w]:.2f}x")

    np.savez("sweep_data.npz",
             e2_gauss=saved["gaussian"].ravel(),
             e2_unif=saved["uniform[-1,1]"].ravel(),
             e2_null=_e2_null[:1_000_000])    # subset for figures (same measure & seed)
    print("\nsaved sweep_data.npz")
