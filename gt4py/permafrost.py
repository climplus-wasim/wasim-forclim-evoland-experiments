#!/usr/bin/env python3
"""Validate wasim-gt's simple permafrost model (wasim_gt.processes.permafrost) against WaSiM grid dumps.

The model runs inside the snow model's run1, after each cell's snow update. So the counters, the thaw
depth and the root depth come from the dump point before the snow call, the snow storages from
after_snow1, and the results are compared with after_snow1. Every step, every active cell, exactly,
including NaN, infinity and nodata patterns.

Neither golden setup switches the model on. A variant of the Rhone template does
(rhone_permafrost.wasim.in, 48 hourly steps), with internal constant grids (alpha 0.05, initial thaw
depth 0.5 m) and a refreeze threshold of 0.5 days, so that snow-free, snow-covered and refreezing cells
all occur within 2 days:
  - [standard_grids]: ThawCoeffPermaFrost and PermafrostThawDepth with readcode = 0, and also
    SnowFreeDaysGrid and SnowCoverDaysGrid (WaSiM stops with error 33 without the latter, snowmod.cc:2548)
  - [permafrost]: method, refreeze days, SWE threshold, statistics file, reduced k_sat (5 entries;
    the setups' sections have only the first 3)

Grids to dump (WASIM_DUMP_GRIDS) at start, the point before snow (after_irrigation in Rhone,
after_interception in Doubs) and after_snow1:
  [zone_grid],ThawCoeffPermaFrost,PermafrostThawDepth,SnowFreeDaysGrid,SnowCoverDaysGrid,root_depth1,
  snowstorage_solid,snowstorage_liquid

The port is a DaCe program on the model cells (wasim_gt.cells): grids are gathered to the active cells,
the program runs, and its outputs are compared with the dump on the same cells.

usage: permafrost.py DUMP_DIR CONTROL_FILE [--pre after_irrigation] [--step-minutes 60]
"""
import argparse
import re
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.permafrost import PermafrostParameters, permafrost

from wasim_dump import Dump

OUTPUTS = ("SnowCoverDaysGrid", "SnowFreeDaysGrid", "PermafrostThawDepth", "root_depth1")


def permafrost_parameters(control_file):
    text = open(control_file, encoding="utf-8", errors="replace").read()
    section = text.split("[permafrost]", 1)[1].split("\n[", 1)[0]
    numbers = []
    for line in section.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and re.fullmatch(r"[-+0-9.eE]+", line):
            numbers.append(float(line))
    if int(numbers[0]) != 1:
        raise SystemExit(f"permafrost method {numbers[0]}: only method 1 exists")
    return PermafrostParameters(refreeze_days=numbers[1], max_swe_snow_free=numbers[2])


def difference(a, b, nodata=-9999.0):
    """Largest absolute difference over the cells; inf if the NaN, infinity or nodata patterns differ."""
    special = lambda x: np.isnan(x) | np.isinf(x) | (x == nodata)
    if not np.array_equal(special(a), special(b)) or not np.array_equal(a[special(a)], b[special(b)], equal_nan=True):
        return float("inf")
    regular = ~special(a)
    return float(np.abs(a[regular] - b[regular]).max()) if regular.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_file", help="the filled control file of the run (run_case.py writes it to the output directory)")
    ap.add_argument("--pre", default="after_irrigation", help="dump point just before the snow model")
    ap.add_argument("--step-minutes", type=float, default=60.0)
    args = ap.parse_args()

    d = Dump(args.dump_dir)
    params = permafrost_parameters(args.control_file)
    active = d("start", "[zone_grid]")[0] > -9999
    cells = CellLayout.from_mask(active)
    fld = cells.to_cells
    alpha = fld(d("start", "ThawCoeffPermaFrost")[0])
    nsteps = d(args.pre, "SnowCoverDaysGrid").shape[0]
    worst = dict.fromkeys(OUTPUTS, 0.0)
    counts = dict(covered=0, thawing=0, refrozen=0, root_limited=0)
    t_call = []
    for t in range(nsteps):
        pre = lambda name: fld(d(args.pre, name)[t])
        post = lambda name: fld(d("after_snow1", name)[t])
        inputs = (post("snowstorage_solid"), post("snowstorage_liquid"), pre("SnowCoverDaysGrid"),
                  pre("SnowFreeDaysGrid"), pre("PermafrostThawDepth"), alpha, pre("root_depth1"))
        start = time.perf_counter()
        res = permafrost(*inputs, params, args.step_minutes)
        t_call.append(time.perf_counter() - start)
        for k, v in zip(OUTPUTS, res, strict=True):
            worst[k] = max(worst[k], difference(v, post(k)))
        cover, thaw = post("SnowCoverDaysGrid"), post("PermafrostThawDepth")
        counts["covered"] += int((cover > 0).sum())
        counts["thawing"] += int((cover == 0).sum())
        counts["refrozen"] += int(((thaw == 0) & (pre("PermafrostThawDepth") > 0)).sum())
        counts["root_limited"] += int((post("root_depth1") != pre("root_depth1")).sum())

    print(params)
    print(f"{cells.n} active cells, {nsteps} steps, DaCe; cell-steps: {counts}")
    print(f"warm call {1000 * float(np.median(t_call[1:])):.3f} ms (median, without the gather)")
    print("largest absolute difference to WaSiM over all steps and active cells:")
    for k, v in worst.items():
        print(f"  {k}: {v:.3g}")
    return 0 if max(worst.values()) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
