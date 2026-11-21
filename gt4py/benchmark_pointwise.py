#!/usr/bin/env python3
"""Warm time per call of the cell-wise DaCe programs of wasim-gt: snow evaporation (both branches),
T-index snow, permafrost and glacier melt, on the cell counts of Doubs (6,022) and Rhone (21,584).

The programs are pointwise, so their time depends on the number of cells, not on the values; the inputs
are random but in the ranges of the validation runs. Two times per program, median of repeated calls:
  - compiled: the compiled SDFG called directly (`program.compile()`), the time of the generated code;
  - call: the `@dace.program` called with numpy arrays, as the wrappers do (adds DaCe's argument handling).
The gather to cells is not included. Run with OMP_NUM_THREADS=1 to compare with WaSiM's one-thread
timings (validation/performance.md).

usage: OMP_NUM_THREADS=1 benchmark_pointwise.py [--repeat 200]
"""
import argparse
import os
import time

import numpy as np

from wasim_gt.processes import glacier_melt, permafrost, snow_evaporation, snow_tindex


def median_ms(f, repeat):
    f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def cases(n, rng):
    """{name: (program, keyword arguments)} for n cells."""
    r = lambda lo=0.0, hi=1.0: rng.uniform(lo, hi, n)
    ssnow = rng.exponential(5.0, n) * (rng.random(n) < 0.7)
    etp, veg = np.array([r(0, 0.3) for _ in range(2)]), np.array([rng.random(n) < 0.9 for _ in range(2)])
    stack = rng.uniform(0, 500, (n, 8))
    out = lambda *shape: np.empty(shape or (n,))
    return {
        "snow evaporation (conventional)": (snow_evaporation.conventional_program, dict(
            ssnow=ssnow, etp=etp, has_vegetation=veg, evap=r(0, 0.1), ssnow_out=out(), etp_out=out(2, n),
            evap_out=out(), evrs_out=out(), N=n, V=2)),
        "snow evaporation (energy balance)": (snow_evaporation.energy_balance_program, dict(
            ssnow=ssnow, etp=etp, has_vegetation=veg, evap=r(0, 0.1), etp_out=out(2, n), evap_out=out(), N=n, V=2)),
        "T-index snow": (snow_tindex.snow_tindex_program, dict(
            temp=r(-5, 8), prec=rng.exponential(1.0, n), ssnow=ssnow, sliq=ssnow * 0.03, no_snow=rng.random(n) < 0.01,
            t0r=1.0, transition=1.0, t0=1.0, c0=1.6, cwh=0.05, steps_per_day=24.0, ssnow_out=out(), sliq_out=out(),
            outflow_out=out(), N=n)),
        "permafrost": (permafrost.permafrost_program, dict(
            ssnow=ssnow, sliq=ssnow * 0.03, snow_cover_days=r(0, 1), snow_free_days=r(0, 5), thaw_depth=r(),
            alpha=r(0, 0.1), root_depth=r(0, 2), refreeze_days=0.5, max_swe=5.0, step_days=1.0 / 24.0,
            cover_out=out(), free_out=out(), thaw_out=out(), root_out=out(), N=n)),
        "glacier melt (all cells glacierized, firn stack of 7 layers)": (glacier_melt.glacier_melt_program, dict(
            temp=r(-2, 10), ssnow_gl=ssnow * (rng.random(n) < 0.3), snow_melt=r(0, 0.3), firn=stack, rad=r(0, 800),
            mf=np.full(n, 2.5), a_ice=np.full(n, 5e-4), a_snow=np.full(n, 2e-4), debris=np.ones(n),
            glacier=np.ones(n, bool), t0=0.0, debris_scaling=1.0, steps_per_day=24.0, qice_out=out(), qfirn_out=out(),
            firn_out=np.empty_like(stack), N=n, K=8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}; median of {args.repeat} warm calls, ms")
    print("| program | Doubs compiled | Doubs call | Rhone compiled | Rhone call |")
    print("| --- | --- | --- | --- | --- |")
    rows = {}
    for n in (6022, 21584):
        for name, (program, kwargs) in cases(n, rng).items():
            compiled = program.compile()
            rows.setdefault(name, []).extend([median_ms(lambda: compiled(**kwargs), args.repeat),
                                              median_ms(lambda: program(**kwargs), args.repeat)])
    for name, v in rows.items():
        print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in v) + " |")


if __name__ == "__main__":
    main()
