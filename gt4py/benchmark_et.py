#!/usr/bin/env python3
"""Warm per-call timing of wasim-gt's radiation correction, potential ET and vegetation parameters.

Takes the inputs of one daylight step from a WaSiM grid dump (as potential_et.py does), gathers
them to the model cells once, and calls each process repeatedly after warm-up calls that include
compilation. Two numbers per process: the Python function (host scalars, argument conversion and
the compiled programs) and the compiled DaCe programs alone (wasim_gt.dace_call.Compiled.seconds);
neither includes compilation, the gather to cells or dump I/O. Compare with WaSiM's per-module
times from WASIM_TIMING (wasim-eth) on the same case and a single thread (OMP_NUM_THREADS=1): see
../validation/performance.md.

usage: benchmark_et.py DUMP_DIR CONTROL_INTERN START [--step T] [--repeat N] [--cellsize M]
"""
import argparse
import datetime
import re
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes import potential_et as pet
from wasim_gt.processes import radiation_correction as rcm
from wasim_gt.processes import vegetation as veg
from wasim_gt.processes.sun import Site

from potential_et import WaSiMTime, multilayer_table, numbers, section
from vegetation import landuse_table
from wasim_dump import Dump


def timed(fn, repeat, programs=(), warmup=2):
    """(ms per call of fn, ms per call in the compiled programs)"""
    for _ in range(warmup):
        fn()
    for prog in programs:
        prog.reset_timer()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    total = 1000.0 * (time.perf_counter() - t) / repeat
    return total, 1000.0 * sum(prog.seconds for prog in programs) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_intern")
    ap.add_argument("start", help="first step of the dump, e.g. 2021-06-13T01")
    ap.add_argument("--step", type=int, default=11, help="dump step to use (default 11: midday of the first day)")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--cellsize", type=float)
    args = ap.parse_args()

    text = open(args.control_intern, encoding="latin-1").read()
    coords, rc, et, lake = (section(text, s) for s in ("coordinates", "radiation_correction", "evapotranspiration",
                                                         "lake_model"))
    step = float(et[1])
    site = Site(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]),
                float(section(text, "output_interval")[3].split()[0]) / 60.0)
    amp = numbers(et[30])
    lakes_on = int(lake[0]) != 0
    params = pet.Parameters(cr=tuple(numbers(et[25])), dt_koeff=numbers(et[28])[0], dtmax=tuple(numbers(et[29])[:12]),
                            ampl_day=amp[0], ampl_annual=amp[1], albedo_open_water=float(lake[2]) if lakes_on else 0.0,
                            z0_open_water=float(lake[3]) if lakes_on else 0.0, lakes=lakes_on)
    d = Dump(args.dump_dir)
    p0, p1, t = "after_meteo", "after_expokorr", args.step
    active = d(p0, "[zone_grid]")[0] > -9999
    dem = np.array(d(p0, "[elevation_model]")[0])
    nrows, ncols = dem.shape
    cellsize = args.cellsize
    if cellsize is None:
        cellsize = float(re.search(r"cellsize\s+([0-9.]+)", open(section(text, "elevation_model")[0], "rb")
                                   .read(400).decode("latin-1")).group(1))
    cells = CellLayout.from_mask(active)
    fld = cells.to_cells
    shading = rcm.Shading.from_grid(cells, dem)
    date, hour = args.start.split("T")
    now = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=int(hour), minutes=step * t)
    if int(hour) == 24:  # daily data: the step of a day is labelled hour 24 of that day
        now = WaSiMTime(now - datetime.timedelta(days=1), 24)

    slope, aspect = fld(d(p0, "slope_angle")[t]), fld(d(p0, "slope_aspect")[t])
    temp0, ssd0 = fld(d(p0, "temperature")[t]), fld(d(p0, "sunshine_duration")[t])
    rc_call = lambda: rcm.radiation_correction(  # noqa: E731
        site, now.day, now.month, now.hour, now.minute, step, int(rc[2]), float(rc[5]), int(rc[11]), shading, slope,
        aspect, cellsize, temp0, ssd0)
    subs, _ = rcm.sub_intervals(site, now.day, now.month, now.hour, now.minute, step, int(rc[11]))
    shade_call = lambda: rcm.shadow(shading, subs[0].azimuth, subs[0].altitude, cellsize)  # noqa: E731

    ml = multilayer_table(text)
    luse0 = d(p1, "landuse")[0]
    kl, ls = np.full(luse0.shape, 0.3), np.full(luse0.shape, 10.0)
    for code, (k, s) in ml.items():
        kl[luse0 == code], ls[luse0 == code] = k, s
    layers = [pet.Layer(landuse=fld(d(p1, f"landuse{n}")[t]), lai=fld(d(p1, f"leaf_area_index{n}")[t]),
                        coverage=fld(d(p1, f"vegetation_coverage_degree{n}")[t]),
                        rsc=fld(d(p1, f"SurfaceCanopyResistance{n}")[t]), rsi=fld(d(p1, f"SurfaceIntercepResistance{n}")[t]),
                        z0=fld(d(p1, f"RoughnessLength{n}")[t])) for n in (1, 2)]
    tk = fld(d(p1, "corr_temperature")[t])
    ins = [fld(d(p1, g)[t]) for g in ("global_radiation", "sunshine_duration", "radiation_correction", "air_humidity",
                                       "wind_speed", "albedo")]
    rest = dict(dhm=fld(dem), rse=fld(d(p1, "SurfaceEvaporationResistance")[t]), layers=layers, k_lambert=fld(kl),
                lai_scale=fld(ls), lake=fld(d(p1, "lake_codes")[t]), pond=fld(d(p1, "ponding_storage")[t]))
    ssd_mean = fld(d(p1, "SSDTemp")[t])

    def et_call():
        state = pet.State(ssd_mean=ssd_mean.copy())
        return pet.potential_et(site, now.day, now.month, now.hour, step, params, tk, tk, *ins, state=state, **rest)

    table = landuse_table(text)
    tables = veg.Table.from_table(table)
    luse = [fld(d(p0, f"landuse{n}")[0]) for n in (1, 2)]
    dhm = fld(dem)
    veg_call = lambda: veg.vegetation_parameters(table, luse, dhm, now.day, now.month, now.year,  # noqa: E731
                                                 tables=tables)

    print(f"{cells.n} active cells ({nrows} x {ncols}), step {t} = {now.day}.{now.month}. hour {now.hour}, DaCe, "
          f"{args.repeat} calls after warm-up; ms per call: Python function / compiled programs alone")
    rows = (("radiation correction", rc_call, (rcm._shadow_program, rcm._correction_program)),
            ("  of which shading walk", shade_call, (rcm._shadow_program,)),
            ("potential ET", et_call, (pet._potential_et_program,)),
            ("vegetation parameters (once a day)", veg_call, (veg._vegetation_program,)))
    for name, fn, progs in rows:
        total, prog = timed(fn, args.repeat, progs)
        print(f"  {name:36s} {total:8.3f} / {prog:8.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
