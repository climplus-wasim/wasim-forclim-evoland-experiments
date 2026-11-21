#!/usr/bin/env python3
"""Validate wasim-gt's dynamic phenology (DP1-DP4) against WaSiM grid dumps.

Drives wasim_gt.processes.dynamic_phenology and vegetation through a run as WaSiM does
(int_tvar_params, Richards version): the daily mean temperature from the interpolated
temperature of every step, and on each day-end step the moving average, the phenology update
and the vegetation parameters. The state is carried from the start date, like WaSiM's, and
compared with the dump after_meteo on every step: forcing, chilling, F*, start of growth, the
MultipleStartTimes stacks, start of leaf fall, daily mean and moving-average temperatures, and
all vegetation parameters (the albedo on day-end steps only; the snow model changes it).

Reference case: rhone_dynphen.wasim.in (four land uses switched to DP1-DP4). The table is read
from the expanded control file as WaSiM reads it (see vegetation.py, also for the positions
in the dumped land-use grids).

Dump these grids at after_meteo (WASIM_DUMP_GRIDS): [zone_grid], [elevation_model], landuse1..n,
temperature, MeanThetaInRootZone, TemperatureAVG, TemperatureMOVAVG7, StartDayOfLeaveFall, and per
layer SumOfForcingUnits, Pheno_start, SumOfChillingUnits, FStar_ForcingThreshold,
MultipleStartTimes and the vegetation grids of vegetation.py. Grids missing from the dump are not
compared (the dumps of the 577-day case are large: they can be made in parts, e.g. the phenology
grids in one run and the vegetation grids in another, each with the inputs [zone_grid],
[elevation_model], landuse1..n, temperature and MeanThetaInRootZone).

The ports are DaCe programs on the model cells (wasim_gt.cells, cells of the zone grid): grids are
gathered to cells, the state is carried on the cells, and compared with the dump on the same cells.
Run with NPY_DISABLE_CPU_FEATURES=X86_V4 (numpy's vectorised exp and log differ from the C library
in the last bit; the initialisation runs in numpy).

usage: dynamic_phenology.py DUMP_DIR CONTROL_INTERN START [--layers N] [--steps N]
   START: first step as in WaSiM's input data, e.g. 2021-01-01T24 (daily) or 2021-01-01T01 (hourly)
"""
import argparse
import datetime
import re
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes import dynamic_phenology as dp
from wasim_gt.processes import vegetation as veg

from vegetation import GRIDS, initial_dem, section, wasim_param, landuse_table as plain_table
from wasim_dump import Dump

METHODS = {"VariableDayCount": 2, "DynamicPhenology_1": 3, "DynamicPhenology_2": 4,
           "DynamicPhenology_3": 5, "DynamicPhenology_4": 6}
INTS = {"DP1_t1_dorm", "JDReset_TStart", "maxStartJDforDP1", "DP2_t0_dorm", "DP2_t1_dorm", "DP4_t0_dorm",
        "DP4_t1_dorm", "DP4_t_xs1", "DP4_SPSLF"}
FIELDS = {"F*": "f_star", "DP1_t1_dorm": "dp1_t1_dorm", "DP1_T_Bf": "dp1_t_bf", "JDReset_TStart": "jd_reset_tstart",
          "maxStartJDforDP1": "max_start_jd", "DP2_t0_dorm": "dp2_t0_dorm", "DP2_t1_dorm": "dp2_t1_dorm",
          "DP2_T_Bf": "dp2_t_bf", "DP2_T_Bc": "dp2_t_bc", "DP2_Par_a": "dp2_par_a", "DP2_Par_b": "dp2_par_b",
          "DP2_Offset_1": "dp2_offset_1", "DP2_Offset_2": "dp2_offset_2", "DP4_t0_dorm": "dp4_t0_dorm",
          "DP4_t1_dorm": "dp4_t1_dorm", "DP4_T0": "dp4_t0", "DP4_T1": "dp4_t1", "DP4_Par_a": "dp4_par_a",
          "DP4_Par_b": "dp4_par_b", "DP4_T_xylstop": "dp4_t_xylstop", "DP4_t_xs1": "dp4_t_xs1",
          "DP4_SPSLF": "dp4_spslf"}


def scalar(record, key):
    """One value after `key =` up to ';', read like stringstream >> int / double."""
    buf = record[record.find(key):]
    text = buf[buf.find("=") + 1:buf.find(";")].strip()
    if key in INTS:
        return int(re.match(r"[-+]?\d+", text).group(0))
    return float(re.match(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", text).group(0))


def landuse_table(text):
    """{position: LandUse} including dynamic phenology records."""
    body = text.split("[landuse_table]", 1)[1]
    count = int(body.split()[0])
    lines = body[body.index(body.split()[0]) + len(body.split()[0]):].splitlines(keepends=True)[1:]
    records, i = [], 0
    for _ in range(count):
        record = lines[i]
        i += 1
        while "}" not in record:
            record += lines[i]
            i += 1
        records.append(record)
    plain = plain_table(text.replace("DynamicPhenology_1", "VariableDayCount").replace(
        "DynamicPhenology_2", "VariableDayCount").replace("DynamicPhenology_3", "VariableDayCount").replace(
        "DynamicPhenology_4", "VariableDayCount"))
    table = {}
    for pos, record in enumerate(records, start=1):
        method = next(m for name, m in METHODS.items() if name in record.split("{")[0] + record[:200])
        lu = plain[pos]
        if method > 2:
            kw = {"method": method, "hredu_dry": scalar(record, "HReduDry")}
            if "StressFactorDynPhen" in record:
                kw["stress_factor"] = scalar(record, "StressFactorDynPhen")
            for key, name in FIELDS.items():
                if key in record:
                    kw[name] = scalar(record, key)
            if method == 5:
                kw["forcing_threshold"] = wasim_param(record, "ForcingThreshold", len(lu.jul_days))
            lu = veg.LandUse(**{**lu.__dict__, "dynamic": dp.Dynamic(**kw)})
        table[pos] = lu
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_intern")
    ap.add_argument("start")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    text = open(args.control_intern, encoding="latin-1").read()
    table = landuse_table(text)
    step = float(section(text, "evapotranspiration")[1])
    hour_offset = float(section(text, "output_interval")[3].split()[0]) / 60.0
    end_year = int(re.search(r"^\s*(\d+)\s*#\s*end year", text, re.M | re.I).group(1)) \
        if re.search(r"^\s*(\d+)\s*#\s*end year", text, re.M | re.I) else None

    d = Dump(args.dump_dir)
    p = "after_meteo"
    active = d(p, "[zone_grid]")[0] > -9999
    cells = CellLayout.from_mask(active)
    fld = cells.to_cells
    dhm = fld(d(p, "[elevation_model]")[0])
    nl = args.layers
    luse = [fld(d(p, f"landuse{n}")[0]) for n in range(1, nl + 1)]
    temp = d(p, "temperature")
    throot = d(p, "MeanThetaInRootZone") if d.has(p, "MeanThetaInRootZone") else None
    nsteps = temp.shape[0] if not args.steps else min(args.steps, temp.shape[0])

    date, hour = args.start.split("T")
    start = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=int(hour))
    hour24 = int(hour) == 24

    def wasim_time(t):
        now = start + datetime.timedelta(minutes=step * t)
        if hour24:
            prev = now - datetime.timedelta(days=1)
            return prev.day, prev.month, prev.year, 24
        return now.day, now.month, now.year, now.hour

    last = wasim_time(nsteps - 1)
    if end_year is None:
        end_year = last[2]
    day0, month0, year0, _ = wasim_time(0)
    dem0 = fld(initial_dem(text, active.shape))  # WaSiM initialises before the lake model lowers lake cells
    state = dp.initial_state(table, luse, dem0, day0, month0, year0, end_year)
    tables = veg.Table.from_table(table)
    params = veg.vegetation_parameters(table, luse, dem0, day0, month0, year0, initial=True, dynamic=state,
                                       tables=tables)

    names = {"forc": "SumOfForcingUnits", "tstart": "Pheno_start", "chill": "SumOfChillingUnits",
             "fstar": "FStar_ForcingThreshold", "mtstart": "MultipleStartTimes"}
    worst, first_bad = {}, {}

    def compare(g, ours, ref_grid, t):
        if not d.has(p, ref_grid[0]):
            return
        ref = d(p, ref_grid[0])[t]
        if ref_grid[1] is not None:
            ref = ref[..., ref_grid[1]]
        diff = np.abs(ours - fld(ref))
        w = float(np.nanmax(diff)) if diff.size else 0.0
        worst[g] = max(worst.get(g, 0.0), w)
        if w > 0 and g not in first_bad:
            first_bad[g] = t

    t_dp, t_veg = [], []
    for t in range(nsteps):
        day, month, year, hr = wasim_time(t)
        t1 = time.perf_counter()
        dp.accumulate_temperature(state, fld(temp[t]), step)
        t_acc = time.perf_counter() - t1
        day_end = veg.is_day_end(hr, 0, hour_offset)
        if day_end:
            th = fld(throot[t]) if throot is not None else None
            t1 = time.perf_counter()
            dp.day_end_update(state, table, luse, day, month, end_year, th)
            t_dp.append(time.perf_counter() - t1 + t_acc)
            t1 = time.perf_counter()
            params = veg.vegetation_parameters(table, luse, dhm, day, month, year, dynamic=state, tables=tables)
            t_veg.append(time.perf_counter() - t1)
        for n in range(nl):
            for key, g in names.items():
                ours = getattr(state, key)
                if ours is None:
                    continue
                compare(f"{g}{n + 1}", ours[n], (f"{g}{n + 1}", None), t)
            for key, g in GRIDS.items():
                compare(f"{g}{n + 1}", params["layers"][n][key], (f"{g}{n + 1}", None), t)
        compare("SurfaceEvaporationResistance", params["rse"], ("SurfaceEvaporationResistance", None), t)
        if day_end:
            compare("albedo", params["albedo"], ("albedo", None), t)
        compare("StartDayOfLeaveFall", state.startfall, ("StartDayOfLeaveFall", None), t)
        compare("TemperatureMOVAVG7", state.tmov, ("TemperatureMOVAVG7", slice(0, 8)), t)
        if day_end:
            dp.reset_temperature_mean(state)
        compare("TemperatureAVG", state.tavg, ("TemperatureAVG", None), t)

    print(f"{cells.n} active cells, {nsteps} steps of {step:g} min from {args.start}, end year {end_year}, DaCe")
    if len(t_dp) > 1:
        print(f"warm ms per day (median, without the gather to cells): phenology {1000 * np.median(t_dp[1:]):.2f}, "
              f"vegetation parameters {1000 * np.median(t_veg[1:]):.2f}")
    methods = sorted({(pos, lu.dynamic.method) for pos, lu in table.items() if lu.dynamic is not None})
    print("dynamic land uses (position, WaSiM method):", methods)
    print("largest absolute difference to WaSiM over all steps and active cells (first step with a difference):")
    for g, v in worst.items():
        print(f"  {g}: {v:.3g}" + (f" (step {first_bad[g]})" if g in first_bad else ""))
    print("compared:", len(worst), "grids")
    return 0 if max(worst.values()) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
