#!/usr/bin/env python3
"""Validate wasim-gt's snow evaporation (wasim_gt.processes.snow_evaporation) against WaSiM grid dumps.

Inputs: dump point after_interception (WaSiM order for interception method 1: snow run1,
interception, snow evaporation; for method 11: interception, snow run1, snow evaporation, so use
--pre after_snow1 there). Reference: after_snowevap, every step and active cell, exactly.

The port is a DaCe program on the model cells (wasim_gt.cells): grids are gathered to cells, the
program runs, and its outputs are compared with the dump on the same cells.

usage: snow_evaporation.py DUMP_DIR [--energy-balance] [--pre POINT]
"""
import argparse
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.snow_evaporation import conventional, energy_balance

from wasim_dump import Dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pre", default="after_interception")
    ap.add_argument("--energy-balance", action="store_true", help="snow method 5, 6, 7 or 11")
    args = ap.parse_args()

    d = Dump(args.dump_dir)
    pre, post = args.pre, "after_snowevap"
    nl = args.layers
    ssnow_in = d(pre, "snowstorage_solid")
    nsteps, nrows, ncols = ssnow_in.shape
    cells = CellLayout.from_mask(ssnow_in[0] > -9999)
    fld = cells.to_cells
    msk = lambda a: cells.to_cells(a, dtype=bool)

    names = ["snowstorage_solid", "potential_evaporation"] + [f"pot_transpiration{l}" for l in range(1, nl + 1)]
    if not args.energy_balance:
        names.append("snow_evaporation")
    worst = dict.fromkeys(names, 0.0)
    t_run, t_call = 0.0, []
    for t in range(nsteps):
        start = time.perf_counter()
        ssnow = fld(ssnow_in[t])
        evap = fld(d(pre, "potential_evaporation")[t])
        etps = [fld(d(pre, f"pot_transpiration{l}")[t]) for l in range(1, nl + 1)]
        veg = [msk(d(pre, f"landuse{l}")[t] > -9999) for l in range(1, nl + 1)]
        call = time.perf_counter()
        if args.energy_balance:
            etps, evap = energy_balance(ssnow, etps, veg, evap)
            res = dict(snowstorage_solid=ssnow, potential_evaporation=evap)
        else:
            ssnow, etps, evap, evrs = conventional(ssnow, etps, veg, evap)
            res = dict(snowstorage_solid=ssnow, potential_evaporation=evap, snow_evaporation=evrs)
        t_call.append(time.perf_counter() - call)
        res.update({f"pot_transpiration{l}": e for l, e in enumerate(etps, start=1)})
        t_run += time.perf_counter() - start
        for name in names:
            diff = np.abs(res[name] - fld(d(post, name)[t]))
            worst[name] = max(worst[name], float(diff.max()))

    print(f"grid {nrows} x {ncols}, {cells.n} active cells, {nsteps} steps, "
          f"{'energy-balance' if args.energy_balance else 'conventional'} snow method, DaCe")
    print(f"{1000 * t_run / nsteps:.2f} ms per step including the gather to cells (first step includes compilation); "
          f"warm call {1000 * float(np.median(t_call[1:])):.3f} ms (median)")
    print("largest absolute difference to WaSiM over all steps and active cells [mm]:")
    for name, v in worst.items():
        print(f"  {name}: {v:.3g}")
    return 0 if max(worst.values()) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
