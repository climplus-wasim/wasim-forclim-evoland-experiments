#!/usr/bin/env python3
"""Warm time per call of the DaCe programs of snow method 7 and the glacier dynamics, on WaSiM dumps.

Times the compiled programs only (no gather to cells, no argument conversion of the wrappers), on the
inputs of one step of the dumps, median of repeated calls. Run with OMP_NUM_THREADS=1 to compare with
WaSiM's single-thread module times (validation/performance.md).

usage: benchmark_snow_glacier.py snow DUMP_DIR CONTROL_FILE [--pre after_interception --input throughfall0]
       benchmark_snow_glacier.py glacier DUMP_DIR CONTROL_FILE --start YYYY-MM-DD
"""
import argparse
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes import glacier_dynamics as gd
from wasim_gt.processes import snow_energy_balance as seb
from wasim_gt.processes import snow_slides as sl

from glacier_dynamics import dynamics_parameters
from snow_energy_balance import landuse_albedo, snow_parameters
from wasim_dump import Dump


def median_ms(fn, repeat=30):
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["snow", "glacier"])
    ap.add_argument("dump_dir")
    ap.add_argument("control_file")
    ap.add_argument("--pre", default="after_interception")
    ap.add_argument("--input", default="throughfall0")
    ap.add_argument("--start", default=None)
    ap.add_argument("--step", type=int, default=12)
    args = ap.parse_args()
    d = Dump(args.dump_dir)
    active = d("start", "[zone_grid]")[0] > -9999
    cells = CellLayout.from_mask(active)
    tc = cells.to_cells
    n = cells.n
    t = args.step
    params, slide = snow_parameters(args.control_file)
    if args.what == "snow":
        g = lambda name: tc(d(args.pre, name)[t])
        lu_alb, _ = landuse_albedo(d, active, args.pre, 1)
        no_snow = ((g("lake_codes") > 0) & (g("ponding_storage") > 0)).astype(float)
        ins = [g("corr_temperature"), g("corr_temperature"), g(args.input), g("deposition_index"), g("air_humidity"),
               g("cloudiness"), g("AspectCorrectedGlobalRadiation"), g("wind_speed"),
               tc(d("start", "[elevation_model]")[0]), tc(lu_alb[t]), no_snow, np.ones(n), g("snowstorage_solid"),
               g("snowstorage_liquid"), g("snowtemp"), g("surftemp"), g("snow_age"),
               *[np.zeros(n) for _ in range(5)], *[np.empty(n) for _ in seb.OUTPUTS]]
        kw = dict(zip(seb.snow_energy_balance_program.argnames, ins, strict=False))
        kw.update(inv=60.0, inv_tag=24.0, transzone=params.transition, t0r=params.t0r, t0=params.t0, cwh=params.cwh,
                  albmin=params.albedo_min, albmax=params.albedo_max, lwincorr=params.lwin_corr,
                  lwoutcorr=params.lwout_corr, enbalscaling=params.enbal_scaling, redistribute=1.0, nodata=-9999.0,
                  onglacier=0.0, anteil=1.0, fixalb=0.0, finish=1.0, addprev=0.0, N=n)
        prog = seb._compiled()
        print(f"snow part 1 (one call, hourly step): {median_ms(lambda: prog(**kw)):.3f} ms, {n} cells")
        nsteps = d("after_snow2", "input_mass").shape[0]
        nsub = d("after_gwflow", "snow_storage_total").shape[0] // nsteps
        last = lambda name: tc(d("after_gwflow", name)[(t + 1) * nsub - 1])
        s0 = lambda name: tc(d("start", name)[0])
        ice = np.clip(s0("GlacierizedCells"), 0.0, 1.0) if d.has("start", "GlacierizedCells") else np.zeros(n)
        glacier = [last(k) for k in ("TotalSnowStorage_onGlacier", "LiquidSnowStorage_onGlacier",
                                     "SolidSnowStorage_onGlacier")] if d.has("start", "GlacierizedCells") else \
            [np.zeros(n)] * 3
        sargs = [last("snow_storage_total"), last("snowstorage_liquid"), last("snowstorage_solid"), ice, *glacier,
                 s0("[zone_grid]"), s0("slope_angle"), np.stack([s0(f"slidefraction{k}") for k in range(1, 5)]),
                 s0("max_deposition"),
                 sl.visiting_order(cells, d("start", "elev_order_row")[0], d("start", "elev_order_col")[0]),
                 cells.neighbours(sl.SLIDE_NEIGHBOURS), slide["min_slide_slope"], slide["slide_mass_fraction"],
                 *[np.empty(n) for _ in range(9)]]
        prog = sl._compiled()
        skw = dict(zip(sl.snow_slides_program.argnames, sargs, strict=True))
        print(f"snow slides: {median_ms(lambda: prog(**skw, N=n)):.3f} ms")
    else:
        (_, _), p = dynamics_parameters(args.control_file)
        s0 = lambda name: tc(d("start", name)[0])
        model = gd.GlacierModel.from_cells(p, 500.0, cells, s0("[elevation_model]"), s0("zonegrid_soilmodel"),
                                           s0("[zone_grid]"), s0("GlacierizedCells"), s0("GlacierCodes"),
                                           tc(d("start", "firn_WE_stack")[0][..., 0]))
        nsteps = d("after_snow2", "GlacierMassBalance").shape[0]
        nsub = d("after_gwflow", "GlacierMassBalance").shape[0] // nsteps
        gw = lambda name: tc(d("after_gwflow", name)[(t + 1) * nsub - 1])
        s1 = lambda name: tc(d("after_snow1", name)[t])
        s2 = lambda name: tc(d("after_snow2", name)[t])
        margs = [gw("GlacierMassBalance"), gw("GlacierizedCells"), s1("snow_fraction"), s1("rain_fraction"),
                 s1("SnowMelt_onGlacier"), s1("melt_from_firn"), s1("melt_from_ice"), gw("SnowEvaporation_onGlacier"),
                 s2("deposition"), s2("mobile_mass"), s2("input_mass"), 1.0, np.empty(n)]
        prog = gd._compiled(gd.mass_balance_program)
        mkw = dict(zip(gd.mass_balance_program.argnames, margs, strict=True))
        print(f"glacier mass balance: {median_ms(lambda: prog(**mkw, N=n)):.3f} ms, {n} cells")
        state = dict(ice=gw("GlacierizedCells"), ice_old=gw("GlacierizedCellsOld"), glacier_id=gw("GlacierCodes"),
                     glacier_id_old=gw("GlacierCodesOld"), gmb=gw("GlacierMassBalance"),
                     gmbo=gw("OldGlacierMassBalance"), firn=tc(d("after_gwflow", "firn_WE_stack")[(t + 1) * nsub - 1]),
                     ssto=gw("snow_storage_total"), ssnow=gw("snowstorage_solid"), sliq=gw("snowstorage_liquid"),
                     ssto_gl=gw("TotalSnowStorage_onGlacier"), ssnow_gl=gw("SolidSnowStorage_onGlacier"),
                     sliq_gl=gw("LiquidSnowStorage_onGlacier"))
        import copy
        fresh = copy.deepcopy(model)
        ms = median_ms(lambda: copy.deepcopy(fresh).yearly_update(state), repeat=5)
        print(f"yearly update (wrapper, including copies of its tables): {ms:.1f} ms, {model.glacier_count} glaciers")


if __name__ == "__main__":
    raise SystemExit(main())
