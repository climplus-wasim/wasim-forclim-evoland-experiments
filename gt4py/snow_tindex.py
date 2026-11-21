#!/usr/bin/env python3
"""GT4Py prototype of WaSiM's T-index snow model (snow method 1), checked against WaSiM output.

What it reimplements (src/snowmod.cc, calc_Accu_and_Melt and run2, hourly time steps):
  1. split precipitation into snow and rain with a linear transition around T0R
     (no snow on lake cells with water in the pond)
  2. accumulate: SSNOW += snow, SLIQ += rain
  3. melt if T > T0 and SSNOW > 0: M = C0 * (T - T0) / steps_per_day, at most SSNOW
  4. move melt to the liquid store; outflow = SLIQ above the holding capacity CWH * SSNOW
  5. subtract snow evaporation (EVRS). WaSiM computes it in the evapotranspiration module
     (etpetr.cc, verdunstungsmodellklasse::snow) after the snow model, from the potential
     evapotranspiration of each vegetation layer and the soil evaporation. The prototype takes
     WaSiM's EVRS grid as an input instead of recomputing it.
Method 1 has no refreezing (WaSiM only refreezes for methods 3 and 4).

Inputs come from a WaSiM run that wrote, at every time step, the corrected temperature
(tcor), precipitation (prec), snow evaporation (etrs) and the snow storages and outflow (ssno, sliq, qsno)
as netCDF; see README.md. The check starts from the same initial states, uses WaSiM's own temperature
and precipitation fields and carries its own snow state through the 240 steps, so any difference comes
from the snow step itself. C0 is interpolated to the day of each step as in WaSiM (the run starts on
2021-01-01, hour 1). In this mode EVRS (the sum over the layers) is subtracted at once; WaSiM takes
it layer by layer, so the result differs by rounding (about 1e-13 mm).

Exact check (--dump DIR): grid dumps of the same run (doubs_snow_tindex.wasim.in) with WASIM_DUMP_POINTS
after_etp,after_interception,after_snow1,after_snowevap and WASIM_DUMP_GRIDS
snowstorage_solid,snowstorage_liquid,potential_evaporation,pot_transpiration1,pot_transpiration2,landuse1,
landuse2,snow_evaporation,corr_temperature,precipitation,lake_codes,ponding_storage,snowcover_outflow.
Every step starts from WaSiM's state before the snow model (after_etp); the snow step is compared with
after_snow1 and, followed by the snow evaporation (wasim_gt.processes.snow_evaporation) on the potential
ET of after_interception, with after_snowevap. Every step and active cell, exactly.

The port is a DaCe program on the model cells (wasim_gt.cells).

usage: snow_tindex.py WASIM_OUTPUT_DIR INITIAL_STATE_DIR LAKE_GRID [--dump DUMP_DIR]
(the netCDF mode needs netCDF4, e.g. `uv run --with netCDF4 python snow_tindex.py ...`)
"""
import argparse
import datetime as dt
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.snow_evaporation import conventional
from wasim_gt.processes.snow_tindex import TIndexParameters, snow_tindex

# parameters of the Doubs setup ([snow_model], method 1); C0 is the same for all months there
PARAMS = TIndexParameters(t0r=1.0, transition=1.0, t0=1.0, c0=(1.6,) * 12, cwh=0.05)
START = dt.datetime(2021, 1, 1)  # date of the run's start; the first step ends at hour 1


def step_date(t, first_hour=1):
    """The date WaSiM uses for step t: that of the step's end, so the step ending at 24:00 is on the next day."""
    return START + dt.timedelta(hours=t + first_hour)


def read_ascii_grid(path):
    with open(path) as f:
        header = {}
        for _ in range(6):
            key, value = f.readline().split()[:2]
            header[key.lower()] = float(value)
        data = np.loadtxt(f)
    return data, header["nodata_value"]


def read_nc_series(path, name):
    import netCDF4
    with netCDF4.Dataset(path) as d:
        # WaSiM writes (time, x, y) with y from north to south; transpose to (time, row, col)
        return np.asarray(d.variables[name][:], dtype="f8").transpose(0, 2, 1)


def check_dump(dump_dir, layers=2):
    """Per step from WaSiM's state: snow step against after_snow1, with snow evaporation against after_snowevap."""
    from wasim_dump import Dump
    d = Dump(dump_dir)
    pre = "after_etp"
    nsteps = d(pre, "snowstorage_solid").shape[0]
    cells = CellLayout.from_mask(d(pre, "snowstorage_solid")[0] > -9999)
    fld = cells.to_cells
    worst = {k: 0.0 for k in ("snowstorage_solid@after_snow1", "snowstorage_liquid", "snowcover_outflow",
                              "snowstorage_solid@after_snowevap", "snow_evaporation")}
    t_call, counts = [], dict(snowfall=0, melt=0, outflow=0)
    for t in range(nsteps):
        g = lambda point, name: fld(d(point, name)[t])
        no_snow = cells.to_cells((d(pre, "lake_codes")[t] > 0) & (d(pre, "ponding_storage")[t] > 0), dtype=bool)
        temp, prec = g(pre, "corr_temperature"), g(pre, "precipitation")
        day = step_date(t)
        start = time.perf_counter()
        ssnow, sliq, outflow = snow_tindex(temp, prec, g(pre, "snowstorage_solid"), g(pre, "snowstorage_liquid"),
                                           no_snow, PARAMS, day.day, day.month)
        t_call.append(time.perf_counter() - start)
        post = "after_snow1"
        for k, v in (("snowstorage_solid@after_snow1", ssnow), ("snowstorage_liquid", sliq), ("snowcover_outflow", outflow)):
            worst[k] = max(worst[k], float(np.abs(v - g(post, k.split("@")[0])).max()))
        mid = "after_interception"
        s2, _, _, evrs = conventional(ssnow, [g(mid, f"pot_transpiration{l}") for l in range(1, layers + 1)],
                                      [fld(d(mid, f"landuse{l}")[t] > -9999, dtype=bool) for l in range(1, layers + 1)],
                                      g(mid, "potential_evaporation"))
        worst["snowstorage_solid@after_snowevap"] = max(worst["snowstorage_solid@after_snowevap"],
                                                        float(np.abs(s2 - g("after_snowevap", "snowstorage_solid")).max()))
        worst["snow_evaporation"] = max(worst["snow_evaporation"],
                                        float(np.abs(evrs - g("after_snowevap", "snow_evaporation")).max()))
        counts["snowfall"] += int(((temp <= PARAMS.t0r + PARAMS.transition) & (prec > 0)).sum())
        counts["melt"] += int(((temp > PARAMS.t0) & (g(pre, "snowstorage_solid") > 0)).sum())
        counts["outflow"] += int((outflow > 0).sum())
    print(f"{cells.n} active cells, {nsteps} steps, DaCe; cell-steps: {counts}")
    print(f"warm call of the snow step {1000 * float(np.median(t_call[1:])):.3f} ms (median)")
    print("largest absolute difference to WaSiM over all steps and active cells [mm]:")
    for k, v in worst.items():
        print(f"  {k}: {v:.3g}")
    return 0 if max(worst.values()) == 0.0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wasim_out")
    ap.add_argument("state_dir")
    ap.add_argument("lake_grid")
    ap.add_argument("--dump", help="grid dumps of the same run: exact per-step check (see the docstring)")
    args = ap.parse_args()
    if args.dump:
        return check_dump(args.dump)

    temp = read_nc_series(f"{args.wasim_out}/tcor.grd.nc", "tcor")
    prec = read_nc_series(f"{args.wasim_out}/prec.grd.nc", "prec")
    evrs = read_nc_series(f"{args.wasim_out}/etrs.grd.nc", "etrs")
    ref_ssno = read_nc_series(f"{args.wasim_out}/ssno.grd.nc", "ssno")
    ref_sliq = read_nc_series(f"{args.wasim_out}/sliq.grd.nc", "sliq")
    ref_qsno = read_nc_series(f"{args.wasim_out}/qsno.grd.nc", "qsno")
    ssno0, nodata = read_ascii_grid(f"{args.state_dir}/ssno.grd")
    sliq0, _ = read_ascii_grid(f"{args.state_dir}/sliq.grd")
    lakes, lake_nodata = read_ascii_grid(args.lake_grid)

    valid = (ssno0 != nodata) & (ref_ssno[0] > -9998)
    lake = (lakes > 0) & (lakes != lake_nodata)
    nsteps, nrows, ncols = temp.shape
    print(f"grid {nrows} x {ncols}, {valid.sum()} active cells, {lake.sum()} lake cells, {nsteps} steps")

    cells = CellLayout.from_mask(valid)
    fld = cells.to_cells
    ssnow, sliq = fld(ssno0), fld(sliq0)
    # WaSiM sets the snow fraction to 0 on lake cells only while their pond holds water;
    # the pond state is not compared here, so lake cells are excluded from the comparison
    no_snow = cells.to_cells(lake, dtype=bool)
    compare = ~no_snow
    worst = {"ssno": 0.0, "sliq": 0.0, "qsno": 0.0}
    t_kernel = 0.0
    for t in range(nsteps):
        start = time.perf_counter()
        day = step_date(t)
        ssnow, sliq, outflow = snow_tindex(fld(temp[t]), fld(prec[t]), ssnow, sliq, no_snow, PARAMS, day.day, day.month)
        e = fld(evrs[t])
        ssnow = ssnow - np.where(e < ssnow, e, ssnow)  # snow evaporation, as a total (see the docstring)
        if t == 0:
            t_first = time.perf_counter() - start
        else:
            t_kernel += time.perf_counter() - start
        for name, mine, ref in zip(worst, (ssnow, sliq, outflow), (ref_ssno[t], ref_sliq[t], ref_qsno[t])):
            worst[name] = max(worst[name], float(np.abs(mine - fld(ref))[compare].max()))

    swe_end = (ssnow + sliq)[compare]
    print(f"DaCe: first step (includes compilation) {t_first:.2f} s, then {1000 * t_kernel / (nsteps - 1):.2f} ms per step")
    print(f"snow water equivalent at the end: mean {swe_end.mean():.3f} mm, max {swe_end.max():.3f} mm")
    print("largest absolute difference to WaSiM over all steps and non-lake cells [mm]:")
    for name, value in worst.items():
        print(f"  {name}: {value:.3g}")
    return 0 if max(worst.values()) <= 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
