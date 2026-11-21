#!/usr/bin/env python3
"""Throughput per ensemble member of wasim-dace's driver programs, members as extra cells (#14,
wasim_gt.driver.ensemble; docs/architecture.md §5), against WaSiM run as separate processes.

Runs the driver chained from WaSiM's state (as chained.py) up to step --step, captures the inputs of the
Richards column and of the cell-wise chain (interception, snow part 1, snow evaporation) at that step, and
times one call of each for E = 1, 4, 16 members (median of --repeat calls, state reset before each):
  - Richards column (unsatzon method 3, WaSiM's row buffer carried): members with the saturated conductivity
    scaled by a factor per member (member 0: 1, the run's own parameters), so that they do different work;
  - the cell-wise chain with the interception capacity scaled per member.
Member 0 is checked against the single run bit for bit. Prints ms per call and per member; run once with
OMP_NUM_THREADS=1 and once with 4.

usage: OMP_NUM_THREADS=1 ensemble_benchmark.py DUMP RUNS [--case doubs] [--step 13] [--members 1,4,16]
"""
import argparse
import copy
import dataclasses
import os
import time

import chained
import numpy as np
import richards as rzv

from wasim_gt.driver import bound
from wasim_gt.driver import ensemble as ens
from wasim_gt.driver import model as dm

DYNAMIC = ("vbg", "wurz", "etp", "prec", "evap", "snow")
PROFILES = ("theta", "hh", "hgeo", "qu")


def capture(setup, step):
    """Run the driver to `step` and return the Richards inputs, the soil state before it, and the inputs of
    the cell-wise chain of that step."""
    state = setup.initial_state()
    model = dm.Model(setup.config, setup.n)
    grab = {}

    def hook(point, st, h):
        if st.step != step:
            return
        if point == "after_etp":
            grab["etp"] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in h.items()}
            grab["state_etp"] = copy.deepcopy(st)
        if point == "after_snowevap":
            grab["h"] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in h.items()}
            grab["soil"] = copy.deepcopy(st.soil.as_dict())
            grab["snow_solid"] = st.snow.solid.copy()
            grab["vegetation"] = copy.deepcopy(st.vegetation)

    for t in range(step + 1):
        f = setup.forcing(t)
        if t == step:
            grab["forcing"] = f
        model.step(state, f, hook)
    return model, grab


def median_ms(fn, repeat):
    fn()  # warm (compiles on the first use)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("runs")
    ap.add_argument("--case", default="doubs")
    ap.add_argument("--step", type=int, default=13)
    ap.add_argument("--members", default="1,4,16")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    setup = chained.Setup(args.dump_dir, args.runs, args.case)
    model, g = capture(setup, args.step)
    n, cfg = setup.n, setup.config
    text = chained.text_of(setup.intern)
    soils = rzv.soil_types(text)
    soil_cfg = cfg.soil
    v = g["vegetation"]
    h = g["h"]
    single_inputs = dict(vbg=v.vcf, wurz=v.root_depth, etp=h["pot_transpiration"],
                         prec=h["snowcover_outflow"] if soil_cfg.prec_from_snow else h["throughfall0"],
                         evap=h["potential_evaporation"], snow=g["snow_solid"])
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"{args.case}: {n} cells, step {args.step}, OMP_NUM_THREADS={threads}")
    print(f"{'members':>8} {'Richards ms/call':>17} {'ms/member':>10} {'cell-wise ms/call':>18} {'ms/member':>10}")
    reference = None
    for members in (int(x) for x in args.members.split(",")):
        factors = [1.0] + [0.5 + k / max(members - 1, 1) for k in range(1, members)]
        member_soils = [[dataclasses.replace(s, ksat=[k * f for k in s.ksat]) for s in soils] for f in factors]
        run = ens.RichardsEnsemble(soil_cfg.options, member_soils, soil_cfg.row_start, soil_cfg.kind,
                                   soil_cfg.static)
        dyn = {k: ens.tile(single_inputs[k], members) for k in DYNAMIC}
        st0 = {k: ens.tile(a, members, axis=0 if k in PROFILES else -1) for k, a in g["soil"].items()}
        work = {}

        def call(run=run, dyn=dyn, st0=st0, work=work):
            work.clear()
            work.update({k: a.copy() for k, a in st0.items()})
            run(dyn, work)

        t_rich = median_ms(call, args.repeat)
        if reference is None:  # member 0 of every ensemble equals the single run
            reference = {k: ens.member(a, 0, n, axis=0 if k in PROFILES else -1).copy() for k, a in work.items()}
        for k, a in work.items():
            if not np.array_equal(ens.member(a, 0, n, axis=0 if k in PROFILES else -1), reference[k],
                                  equal_nan=True):
                raise SystemExit(f"member 0 differs from the single run: {k}")

        # cell-wise chain: interception -> snow part 1 -> snow evaporation, interception capacity per member
        e_et, s_et = g["etp"], g["state_etp"]
        f = g["forcing"]
        cap = ens.stack_members([cfg.interception_capacity * fc for fc in factors])
        icpt = bound.Interception(n * members, cap.shape[0], cap)
        snow = bound.SnowEnergyBalance(n * members, cfg.snow.params, cfg.step_minutes)
        sevap = bound.SnowEvaporation(n * members, cap.shape[0])
        T = lambda a: ens.tile(a, members)  # noqa: E731
        ice = np.zeros(n * members)
        sv = s_et.vegetation
        lai, vcf, lu = T(sv.lai), T(sv.vcf), T(sv.landuse)
        etp, storage = T(e_et["pot_transpiration"]), T(s_et.interception)
        prec = T(f.precipitation)
        no_snow = T(((cfg.et.lake > 0) & (s_et.soil.pond > 0)).astype(float))
        sn = s_et.snow
        snow_in = dict(t=T(e_et["corr_temperature"]), tday=T(e_et["corr_temperature"]), depo=T(cfg.snow.deposition_index),
                       rh=T(f.air_humidity), cl=T(e_et["cloudiness"]), rg=T(e_et["AspectCorrectedGlobalRadiation"]),
                       u=T(f.wind_speed), dhm=T(cfg.snow.elevation), lu_albedo=T(f.landuse_albedo), no_snow=no_snow,
                       ssnow=T(sn.solid), sliq=T(sn.liquid), snowtemp=T(sn.temperature),
                       surftemp=T(sn.surface_temperature), snow_age=T(sn.age))
        evap = T(e_et["potential_evaporation"])

        def chain():  # noqa: B023 (called within the iteration)
            si, ei, q, etp2 = icpt(prec, ice, etp, lai, vcf, lu, storage)
            b = snow(p=q[0], **snow_in)
            sevap(b["ssnow_out"], etp2[1:], lu, evap)

        t_chain = median_ms(chain, args.repeat * 4)
        print(f"{members:>8} {t_rich:>17.2f} {t_rich / members:>10.2f} {t_chain:>18.3f} {t_chain / members:>10.3f}")


if __name__ == "__main__":
    main()
