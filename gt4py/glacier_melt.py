#!/usr/bin/env python3
"""Validate wasim-gt's glacier melt (snow on the glacier fraction and ice/firn melt, glacier method 13)
against WaSiM grid dumps.

Per step, from the dump point before the snow model (after_irrigation in Rhone) to after_snow1:
  1. snow_energy_balance on all cells (the fraction without ice), as in snow_energy_balance.py, now
     compared on glacierized cells too
  2. snow_energy_balance(on_glacier=True) on cells with a glacier fraction, with the first call's
     precipitation, snow and rain fractions and albedo
  3. glacier_melt: ice and firn melt and the firn stack
Every compared grid must match exactly, including NaN, infinity and nodata patterns.

The golden Rhone run (January) never melts ice: every glacier carries snow. A variant runs 1-3 July
2021 from a copy of the Rhone states with the snow on glaciers set to 0 (ssno_onGlac_, sliq_onGlac_,
ssto_onGlac_), so that ice and firn melt (rhone_glacier_melt.wasim.in). A second variant adds a constant
debris grid (internal standard grid debris_on_glaciers, readcode = 0, value 0.7) and a global debris
scaling factor of 0.9 to reach the debris factor (rhone_glacier_melt_debris.wasim.in).

All three steps are DaCe programs on the model cells (wasim_gt.cells): grids are gathered to the active cells,
the programs run on all of them (step 2 with `active` set on the glacierized cells only), and their outputs are
compared with the dump on the active (step 1) or glacierized cells (steps 2 and 3).

Grids to dump at start, after_meteo, after_irrigation and after_snow1: those of snow_energy_balance.py,
plus GlacierizedCells, global_radiation, firn_WE_stack, melt_from_ice, melt_from_firn,
SolidSnowStorage_onGlacier, LiquidSnowStorage_onGlacier, TotalSnowStorage_onGlacier, snowtemp_onGlacier,
surftemp_onGlacier, SnowEvaporation_onGlacier, SnowMelt_onGlacier.

usage: glacier_melt.py DUMP_DIR CONTROL_FILE --start YYYY-MM-DD [--first-hour 1] [--pre after_irrigation]
                       [--input precipitation]
"""
import argparse
import datetime as dt
import re
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.glacier_melt import GlacierMeltParameters, glacier_melt, radiation_coefficient
from wasim_gt.processes.snow_energy_balance import snow_energy_balance

from snow_energy_balance import control_section, difference, landuse_albedo, snow_parameters
from wasim_dump import Dump

GLACIER_SNOW = {"snowstorage_solid": "SolidSnowStorage_onGlacier", "snowstorage_liquid": "LiquidSnowStorage_onGlacier",
                "snow_storage_total": "TotalSnowStorage_onGlacier", "snowtemp": "snowtemp_onGlacier",
                "surftemp": "surftemp_onGlacier", "snow_evaporation": "SnowEvaporation_onGlacier",
                "snowcover_outflow": "SnowMelt_onGlacier"}
FIRST_CALL = ("snowstorage_solid", "snowstorage_liquid", "snow_storage_total", "snowtemp", "surftemp",
              "snow_age", "albedo", "snowcover_outflow", "melt_from_snowcover", "days_snow", "snow_evaporation")


def ice_firn_parameters(control_file):
    """[ice_firn] numbers in reading order: method, t-index factors (ice, firn, snow), melt factor MF,
    radiation coefficients (ice min, ice max, snow min, snow max), ... One value per line is supported."""
    numbers = []
    for line in control_section(control_file, "ice_firn").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and re.fullmatch(r"[-+0-9.eE\s]+", line):
            numbers.append([float(v) for v in line.split()])
    if int(numbers[0][0]) != 13:
        raise SystemExit(f"glacier method {numbers[0][0]}: this check covers method 13 only")
    if any(len(n) != 1 for n in numbers[1:9]):
        raise SystemExit("per-subbasin glacier parameters are not supported by this check")
    v = [n[0] for n in numbers[1:9]]
    return dict(mf=v[3], ice_min=v[4], ice_max=v[5], snow_min=v[6], snow_max=v[7])


def debris_scaling(control_file):
    """The global debris scaling factor of [ice_firn] (the numeric line whose comment mentions the debris grid)."""
    for line in control_section(control_file, "ice_firn").splitlines():
        value, _, comment = line.partition("#")
        if "debris" in comment and re.fullmatch(r"\s*[-+0-9.eE]+\s*", value):
            return float(value)
    raise SystemExit("no debris scaling factor in [ice_firn]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_file")
    ap.add_argument("--start", required=True, help="date of the first step, YYYY-MM-DD")
    ap.add_argument("--first-hour", type=int, default=1)
    ap.add_argument("--pre", default="after_irrigation")
    ap.add_argument("--input", default="precipitation")
    ap.add_argument("--step-minutes", type=float, default=60.0)
    args = ap.parse_args()

    d = Dump(args.dump_dir)
    snow, _ = snow_parameters(args.control_file)
    glac = ice_firn_parameters(args.control_file)
    debris = np.ones(d("start", "[zone_grid]")[0].shape)
    scaling = 1.0
    if d.has("start", "debris_on_glaciers"):  # WaSiM reads the scaling factor only when the grid exists
        grid = d("start", "debris_on_glaciers")[0]
        debris = np.where(grid > -9999 + 1, grid, 1.0)
        scaling = debris_scaling(args.control_file)
    params = GlacierMeltParameters(t0=snow.t0, debris_scaling=scaling)
    active = d("start", "[zone_grid]")[0] > -9999
    lu_alb, unobserved = landuse_albedo(d, active, args.pre, args.first_hour)
    nsteps = d(args.pre, "snowstorage_solid").shape[0]
    cells = CellLayout.from_mask(active)
    fld = cells.to_cells
    elevation = fld(d("start", "[elevation_model]")[0])
    all_cells = np.ones(cells.n, dtype=bool)
    start = dt.datetime.fromisoformat(args.start)
    t_call = []

    worst = {}
    def check(name, value, ref, cells):
        worst[name] = max(worst.get(name, 0.0), difference(value, ref, cells))

    counts = dict(ice_melt=0, firn_melt=0)
    for t in range(nsteps):
        pre = lambda name: fld(d(args.pre, name)[t])
        post = lambda name: fld(d("after_snow1", name)[t])
        glacier = active & (d(args.pre, "GlacierizedCells")[t] > 0)  # on the grid, for the summary
        on_glacier = pre("GlacierizedCells") > 0
        no_snow = ((pre("lake_codes") > 0) & (pre("ponding_storage") > 0)).astype(float)
        base = dict(temperature=pre("corr_temperature"), deposition_index=pre("deposition_index"),
                    air_humidity=pre("air_humidity"), cloudiness=pre("cloudiness"),
                    radiation=pre("AspectCorrectedGlobalRadiation"), wind_speed=pre("wind_speed"), elevation=elevation,
                    landuse_albedo=fld(lu_alb[t]), no_snow=no_snow)
        first = snow_energy_balance(dict(base, precipitation=pre(args.input), active=np.ones(cells.n),
                                         snowstorage_solid=pre("snowstorage_solid"),
                                         snowstorage_liquid=pre("snowstorage_liquid"), snowtemp=pre("snowtemp"),
                                         surftemp=pre("surftemp"), snow_age=pre("snow_age")),
                                    snow, args.step_minutes)
        for k in FIRST_CALL:
            check(k, first[k], post(k), all_cells)
        gl = snow_energy_balance(dict(base, precipitation=first["precipitation"], active=on_glacier.astype(float),
                                      snowstorage_solid=pre(GLACIER_SNOW["snowstorage_solid"]),
                                      snowstorage_liquid=pre(GLACIER_SNOW["snowstorage_liquid"]),
                                      snowtemp=pre(GLACIER_SNOW["snowtemp"]), surftemp=pre(GLACIER_SNOW["surftemp"]),
                                      snow_age=pre("snow_age"), snow_fraction=first["snow_fraction"],
                                      rain_fraction=first["rain_fraction"], albedo_used=first["albedo_used"]),
                                 snow, args.step_minutes, on_glacier=True)
        for k, name in GLACIER_SNOW.items():
            check(name, gl[k], post(name), on_glacier)

        # the date of the step's end time: at 24:00 WaSiM already uses the next day (checked against the dumps)
        day = start + dt.timedelta(hours=t + args.first_hour)
        a_ice = radiation_coefficient(glac["ice_min"], glac["ice_max"], day.day, day.month)
        a_snow = radiation_coefficient(glac["snow_min"], glac["snow_max"], day.day, day.month)
        inputs = (pre("corr_temperature"), gl["snowstorage_solid"], gl["melt"], pre("firn_WE_stack"),
                  pre("global_radiation"), glac["mf"], a_ice, a_snow, fld(debris), on_glacier)
        t0 = time.perf_counter()
        qice, qfirn, stack = glacier_melt(*inputs, params, args.step_minutes)
        t_call.append(time.perf_counter() - t0)
        check("melt_from_ice", qice, post("melt_from_ice"), on_glacier)
        check("melt_from_firn", qfirn, post("melt_from_firn"), on_glacier)
        ref_stack = post("firn_WE_stack")
        for k in range(stack.shape[1]):
            check(f"firn_WE_stack[{k}]", stack[:, k], ref_stack[:, k], on_glacier)
        counts["ice_melt"] += int((post("melt_from_ice")[on_glacier] > 0).sum())
        counts["firn_melt"] += int((post("melt_from_firn")[on_glacier] > 0).sum())

    print(f"{cells.n} active cells, {int(glacier.sum())} glacierized, {nsteps} steps, snow and glacier melt in DaCe, "
          f"{counts}, "
          f"{unobserved} cell-steps with an unobserved land-use albedo")
    print(f"glacier melt: warm call {1000 * float(np.median(t_call[1:])):.3f} ms (median, without the gather)")
    print("largest absolute difference to WaSiM (inf: NaN or nodata patterns differ):")
    for k, v in worst.items():
        print(f"  {k}: {v:.3g}")
    return 0 if max(worst.values()) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
