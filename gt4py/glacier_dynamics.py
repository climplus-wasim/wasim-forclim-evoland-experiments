#!/usr/bin/env python3
"""Validate wasim-gt's glacier mass balance and yearly glacier dynamics (wasim_gt.processes.glacier_dynamics)
against WaSiM grid dumps.

  - every step: the glacier mass balance GMB, from the last sub step's after_gwflow plus this step's
    snow, melt, slide and evaporation grids, compared with after_snow2
  - the evaluation step: the slides (snow_slides, validated in snow_energy_balance.py), the mass balance,
    then the yearly update with the glacier bodies set up from the start dump, compared with after_snow2:
    glacier fraction and codes (and their old values), mass balance and its reference, firn stack, and
    the snow storages off and on the glacier fraction
Every compared grid must match exactly, including NaN, infinity and nodata patterns.

The ports are DaCe programs on the model cells (wasim_gt.cells, row-major): grids are gathered to cells,
the programs run, and their outputs are compared with the dumps on the same cells.

The evaluation runs at 24:00 of the evaluation day of [ice_firn]. WaSiM tests the hour with an offset of 1
(60 * hour + 60 * hour_offset == 1440), so it is the step that ends at 23:00. The golden Rhone run (January)
does not reach it; the Rhone summer variant gt4py/rhone_glacier_summer.wasim.in (1-3 July 2021, evaluation
on 1 July, melt factor 100-fold so that the glaciers retreat) runs on a copy of the Rhone case whose
StateIni_2021 has the snow on glaciers set to 0 (every value other than -9999 in ssno_onGlac_.grd,
sliq_onGlac_.grd and ssto_onGlac_.grd; zero_glacier_snow below does it).

Grids to dump at start, after_snow1, after_gwflow and after_snow2: [zone_grid], zonegrid_soilmodel,
[elevation_model], GlacierizedCells, GlacierizedCellsOld, GlacierCodes, GlacierCodesOld, GlacierMassBalance,
OldGlacierMassBalance, firn_WE_stack, snowstorage_solid, snowstorage_liquid, snow_storage_total,
SolidSnowStorage_onGlacier, LiquidSnowStorage_onGlacier, TotalSnowStorage_onGlacier, snow_fraction,
rain_fraction, SnowMelt_onGlacier, melt_from_firn, melt_from_ice, SnowEvaporation_onGlacier, input_mass,
mobile_mass, deposition, slope_angle, slidefraction1..4, max_deposition, elev_order_row, elev_order_col.

usage: glacier_dynamics.py DUMP_DIR CONTROL_FILE --start YYYY-MM-DD [--first-hour 1] [--cellsize 500]
       glacier_dynamics.py --zero-glacier-snow STATE_DIR   (prepare the summer variant's initial states)
"""
import argparse
import copy
import datetime as dt
import os
import re
import sys
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.glacier_dynamics import DynamicsParameters, GlacierModel, mass_balance_step
from wasim_gt.processes.snow_slides import SLIDE_NEIGHBOURS, snow_slides, visiting_order

from snow_energy_balance import control_section, difference, snow_parameters
from wasim_dump import Dump

COMPARED = ("GlacierizedCells", "GlacierizedCellsOld", "GlacierCodes", "GlacierCodesOld", "GlacierMassBalance",
            "OldGlacierMassBalance", "snow_storage_total", "snowstorage_solid", "snowstorage_liquid",
            "TotalSnowStorage_onGlacier", "SolidSnowStorage_onGlacier", "LiquidSnowStorage_onGlacier")


def dynamics_parameters(control_file):
    """Evaluation date, the volume-area line and the new-glacier line of [ice_firn]."""
    eval_date = va = new = None
    for line in control_section(control_file, "ice_firn").splitlines():
        value = line.split("#", 1)[0].split()
        comment = line.partition("#")[2]
        if "month and day" in comment:
            eval_date = (int(value[0]), int(value[1]))
        elif "VAscaling" in comment:
            va = value
        elif len(value) == 2 and re.fullmatch(r"[0-9.]+", value[0]) and "SWE" in comment:
            new = value
    params = DynamicsParameters(va_scaling=float(va[0]), va_exponent=float(va[1]), elevation_bands=int(va[2]),
                                extra_weight_band0=int(va[3]), max_snow_new_glacier=float(new[0]),
                                rel_size_new_glacier=float(new[1]))
    return eval_date, params


def zero_glacier_snow(state_dir):
    """Set the snow on glaciers to 0 in the ASCII state grids of a copy of StateIni_2021 (keeps -9999)."""
    for name in ("ssno_onGlac_.grd", "sliq_onGlac_.grd", "ssto_onGlac_.grd"):
        path = os.path.join(state_dir, name)
        lines = open(path, newline="").read().split("\r\n")
        out = [ln if i < 6 or not ln.strip() else
               "\t".join(v if float(v) == -9999 else "0" for v in ln.split("\t")) for i, ln in enumerate(lines)]
        open(path, "w", newline="").write("\r\n".join(out))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--zero-glacier-snow":
        zero_glacier_snow(sys.argv[2])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_file")
    ap.add_argument("--start", required=True, help="date of the first step, YYYY-MM-DD")
    ap.add_argument("--first-hour", type=int, default=1)
    ap.add_argument("--cellsize", type=float, default=500.0)
    args = ap.parse_args()

    d = Dump(args.dump_dir)
    (month, day), params = dynamics_parameters(args.control_file)
    _, slide = snow_parameters(args.control_file)
    zone_grid = d("start", "[zone_grid]")[0]
    active = zone_grid > -9999
    cells = CellLayout.from_mask(active)  # row-major, as the glacier walks need
    tc = cells.to_cells
    for name in ("GlacierCodes", "GlacierizedCells"):
        if (d("start", name)[0][~active] > -9999).any():
            raise SystemExit(f"{name} has values outside the model cells: they would be lost on the cell layout")
    s0 = lambda name: tc(d("start", name)[0])
    zone, zone_routing, dhm = s0("[zone_grid]"), s0("zonegrid_soilmodel"), s0("[elevation_model]")
    start_time = time.perf_counter()
    model = GlacierModel.from_cells(params, args.cellsize, cells, dhm, zone_routing, zone, s0("GlacierizedCells"),
                                    s0("GlacierCodes"), tc(d("start", "firn_WE_stack")[0][..., 0]))
    t_startup = time.perf_counter() - start_time
    static = dict(zone=zone, slope=s0("slope_angle"), slide_fractions=[s0(f"slidefraction{k}") for k in range(1, 5)],
                  deposition_max=s0("max_deposition"),
                  order=visiting_order(cells, d("start", "elev_order_row")[0], d("start", "elev_order_col")[0]),
                  neighbours=cells.neighbours(SLIDE_NEIGHBOURS), **slide)
    nsteps = d("after_snow2", "GlacierMassBalance").shape[0]
    nsub = d("after_gwflow", "GlacierMassBalance").shape[0] // nsteps
    start = dt.datetime.fromisoformat(args.start)
    worst = {}
    all_cells = np.ones(cells.n, dtype=bool)

    def check(name, value, ref):
        worst[name] = max(worst.get(name, 0.0), difference(value, ref, all_cells))

    evaluated, t_gmb, t_yearly = [], [], []
    for t in range(nsteps):
        gw = lambda name: tc(d("after_gwflow", name)[(t + 1) * nsub - 1])
        s1 = lambda name: tc(d("after_snow1", name)[t])
        s2 = lambda name: tc(d("after_snow2", name)[t])
        ice = gw("GlacierizedCells")
        args_gmb = (gw("GlacierMassBalance"), ice, s1("snow_fraction"), s1("rain_fraction"), s1("SnowMelt_onGlacier"),
                    s1("melt_from_firn"), s1("melt_from_ice"), gw("SnowEvaporation_onGlacier"), s2("deposition"),
                    s2("mobile_mass"), s2("input_mass"))
        t0 = time.perf_counter()
        gmb = mass_balance_step(*args_gmb)
        t_gmb.append(time.perf_counter() - t0)
        end = start + dt.timedelta(hours=t + args.first_hour)
        if not (end.month == month and end.day == day and end.hour == 23):
            check("GlacierMassBalance (every step)", gmb, s2("GlacierMassBalance"))
            continue
        evaluated.append(t)
        glacier = tuple(gw(k) for k in ("TotalSnowStorage_onGlacier", "LiquidSnowStorage_onGlacier",
                                        "SolidSnowStorage_onGlacier"))
        (ssto, sliq, ssnow), _, (ssto_gl, sliq_gl, ssnow_gl) = snow_slides(
            gw("snow_storage_total"), gw("snowstorage_liquid"), gw("snowstorage_solid"), ice=ice, glacier=glacier,
            **static)
        state = dict(ice=ice, ice_old=gw("GlacierizedCellsOld"), glacier_id=gw("GlacierCodes"),
                     glacier_id_old=gw("GlacierCodesOld"), gmb=gmb, gmbo=gw("OldGlacierMassBalance"),
                     firn=tc(d("after_gwflow", "firn_WE_stack")[(t + 1) * nsub - 1]), ssto=ssto, ssnow=ssnow,
                     sliq=sliq, ssto_gl=ssto_gl, ssnow_gl=ssnow_gl, sliq_gl=sliq_gl)
        before = copy.deepcopy(model)
        t0 = time.perf_counter()
        new = model.yearly_update(state)
        t_yearly.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        before.yearly_update(state)  # again from the same state: the warm time
        t_yearly.append(time.perf_counter() - t0)
        result = dict(GlacierizedCells=new["ice"], GlacierizedCellsOld=new["ice_old"],
                      GlacierCodes=new["glacier_id"], GlacierCodesOld=new["glacier_id_old"],
                      GlacierMassBalance=new["gmb"], OldGlacierMassBalance=new["gmbo"],
                      snow_storage_total=new["ssto"], snowstorage_solid=new["ssnow"], snowstorage_liquid=new["sliq"],
                      TotalSnowStorage_onGlacier=new["ssto_gl"], SolidSnowStorage_onGlacier=new["ssnow_gl"],
                      LiquidSnowStorage_onGlacier=new["sliq_gl"])
        for k in COMPARED:
            check(k, result[k], s2(k))
        ref_firn = tc(d("after_snow2", "firn_WE_stack")[t])
        for k in range(new["firn"].shape[-1]):
            check(f"firn_WE_stack[{k}]", new["firn"][:, k], ref_firn[:, k])
        changed = int((new["ice"] != ice).sum())
        print(f"evaluation at step {t}: {model.glacier_count} glaciers, glacier fraction changed in {changed} cells")

    warm = lambda ts: 1000 * float(np.median(ts[1:])) if len(ts) > 1 else float("nan")
    print(f"{cells.n} active cells, {nsteps} steps, evaluation steps {evaluated}, DaCe; mass balance "
          f"{warm(t_gmb):.3f} ms per step (warm), start-up {1000 * t_startup:.0f} ms (first call, with compilation), "
          f"yearly update {1000 * t_yearly[-1] if t_yearly else float('nan'):.1f} ms (warm)")
    print("largest absolute difference to WaSiM (inf: NaN or nodata patterns differ):")
    for k, v in worst.items():
        print(f"  {k}: {v:.3g}")
    return 0 if evaluated and max(worst.values()) == 0.0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
