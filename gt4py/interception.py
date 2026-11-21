#!/usr/bin/env python3
"""Validate wasim-gt's interception (wasim_gt.processes.interception) against WaSiM grid dumps.

The inputs are the states WaSiM had just before the interception call (dump point after_etp,
because irrigation is off); the results are compared with the dump after_interception, for every
step and active cell. Methods 1 and 11, several vegetation layers. See the module docstring in
wasim-gt for what is ported.

The port is a DaCe program on the model cells (wasim_gt.cells): grids are gathered to cells, the
program runs, and its outputs are compared with the dump on the same cells.

usage: interception.py DUMP_DIR CONTROL_FILE
"""
import argparse
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.interception import Layer, interception

from wasim_dump import Dump, landuse_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_file", help="the filled control file of the run (run_case.py writes it to the output directory)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--sidicke", type=float, default=0.35)
    args = ap.parse_args()

    d = Dump(args.dump_dir)
    pre, post = "after_etp", "after_interception"
    nl = args.layers
    cap_table = landuse_table(args.control_file, "IntercepCap")
    prec = d(pre, "precipitation")
    nsteps, nrows, ncols = prec.shape
    luse = [d(pre, f"landuse{l}") for l in range(1, nl + 1)]
    has_veg = [u > -9999 for u in luse]
    active = d(post, "interception_storage0")[0] > -9999

    def cap_grid(lu):
        c = np.full(lu.shape, args.sidicke)
        for code, v in cap_table.items():
            if v >= 0:
                c[lu == code] = v
        return c

    cells = CellLayout.from_mask(active)
    fld = cells.to_cells
    msk = lambda a: cells.to_cells(a, dtype=bool)
    ice = np.zeros(cells.n)  # no glaciers in the Doubs setup

    names = {"ei": "interception_evaporation", "q": "throughfall", "si": "interception_storage", "etp": "pot_transpiration"}
    worst = {(k, l): 0.0 for k in names for l in range(nl + 1)}
    t_run = 0.0
    for t in range(nsteps):
        start = time.perf_counter()
        layers = [Layer(lai=fld(d(pre, f"leaf_area_index{l}")[t]),
                        coverage=fld(d(pre, f"vegetation_coverage_degree{l}")[t]),
                        capacity=fld(cap_grid(luse[l - 1][t])),
                        etp=fld(d(pre, f"pot_transpiration{l}")[t]),
                        storage=fld(d(pre, f"interception_storage{l}")[t]),
                        has_vegetation=msk(has_veg[l - 1][t]))
                  for l in range(1, nl + 1)]
        res = interception(fld(prec[t]), layers, fld(d(pre, "pot_transpiration0")[t]), ice)
        t_run += time.perf_counter() - start
        for (k, l) in worst:
            ref = d(post, f"{names[k]}{l}")[t]
            diff = np.abs(res[l][k] - fld(ref))
            worst[(k, l)] = max(worst[(k, l)], float(diff.max()))

    print(f"grid {nrows} x {ncols}, {cells.n} active cells, {nl} vegetation layers, {nsteps} steps, DaCe")
    print(f"{1000 * t_run / nsteps:.2f} ms per step including the gather to cells (first step includes compilation)")
    print("largest absolute difference to WaSiM over all steps and active cells [mm]:")
    for (k, l), v in worst.items():
        print(f"  {names[k]}{l}: {v:.3g}")
    return 0 if max(worst.values()) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
