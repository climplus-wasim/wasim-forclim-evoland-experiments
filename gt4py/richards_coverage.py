#!/usr/bin/env python3
"""Which in-scope lines of WaSiM's Richards zone do the reference cases execute, and which of them are ported?

The line set (functions and exclusions) is the one of gt4py/richards.md §5 (design branch, draft PR
27, gt4py/richards/coverage.py), unchanged. On top of it, NOT_PORTED lists the in-scope ranges the DaCe
port (wasim_gt.soil.richards, #7) does not reproduce; every executed in-scope line outside them is
covered by the port and checked bit for bit by gt4py/richards.py on the same 48-step runs.

Build without OpenMP and with coverage (validation/wasim_coverage.py), run each 48-step case with a fresh
set of counters and keep a copy of the object directory per case:

  make MPI=0 DEBUG=1 ARCH=--coverage OMPFLAGS= OMPLIBS= CXXTAG=cov TARGET=wasim-cov
  python3 gt4py/richards_coverage.py WASIM_SRC DOUBS_OBJECTS RHONE_OBJECTS

usage: richards_coverage.py SRC_DIR DOUBS_OBJECT_DIR RHONE_OBJECT_DIR [--lines]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "validation"))
from wasim_coverage import function_range, gcov_lines  # noqa: E402

FUNCTIONS = [
    "pre_run",
    "run",
    "post_run",
    "quflow",
    "gradient_restriction",
    "GetMeanTheta",
    "evapotranspiration",
    "hydraulic_heads_iterative",
    "calc_surface_interaction",
    "specific_fluxes",
    "flux_restrictions_opt",
    "new_thetas",
    "groundwater_recharge",
    "CalculateMacroporeInfiltration",
]
# Inclusive exclusions; all other lines within each named function are retained.
EXCLUSIONS = {
    "pre_run": [
        (7767, 7848, "#9 heat/layered snow"),
        (7897, 7897, "tracer statistics"),
        (7899, 7902, "#9 heat statistics"),
        (7905, 7910, "#8 river table and debug"),
    ],
    "run": [
        (8077, 8092, "#8 ordered river-link gather"),
        (8111, 8247, "row allocation; audited separately"),
        (8357, 8472, "ArcVeg and #9 layered snow"),
        (8476, 8716, "tracers, layered snow, heat and debug"),
        (8818, 8988, "legacy diagnostic dump/replay"),
        (9047, 9095, "tracer groundwater mixing"),
        (9130, 9141, "tracer decay/mixing"),
        (9171, 9189, "tracer mixing"),
        (9359, 9367, "debug tracer balance"),
        (9375, 9380, "debug warning"),
        (9393, 9401, "debug tracer balance"),
        (9416, 9422, "debug-only correction"),
        (9458, 9463, "legacy uncoupled baseflow"),
        (9527, 9531, "#13 irrigation"),
        (9674, 9855, "#9 heat/layered snow"),
        (9897, 9901, "unoptimized flux restrictions: >=200 layers or layered snow"),
        (9914, 9920, "debug output"),
        (9924, 9949, "#9 layered snow"),
        (9955, 10430, "diagnostic dump output and #8 river-table update"),
        (10445, 10462, "#9 layered snow"),
        (10579, 10582, "#13 irrigation"),
        (10625, 10666, "tracer transport"),
        (10674, 10686, "tracer mixing"),
        (10711, 10728, "tracer mixing"),
        (10736, 10742, "tracer clearing"),
        (10801, 10888, "#9 layered snow and ArcVeg"),
        (10904, 11158, "#9 heat-derived grids"),
        (11170, 11199, "#13 irrigation diagnostics"),
        (11226, 11318, "debug balance"),
        (11326, 11381, "#9 heat and layered snow writeback"),
        (11411, 11554, "#9 glacier temperature neighbour search"),
        (11557, 11740, "control-cell diagnostics, deallocation and #8 scatter"),
        (11743, 11890, "HRU spin-up"),
        (11937, 11973, "tracer statistics"),
        (12000, 12016, "uncoupled baseflow"),
        (12033, 12115, "tracer statistics"),
        (12117, 12124, "#9 heat statistics"),
        (12206, 12296, "buffered diagnostic output"),
    ],
    "post_run": [
        (13110, 13116, "ArcVeg and #8 river table"),
        (13149, 13152, "tracers"),
        (13158, 13162, "tracers"),
        (13176, 13179, "tracers"),
        (13185, 13189, "tracers"),
        (13225, 13402, "errors, statistics and legacy output"),
    ],
    "gradient_restriction": [(n, n, "#9 layered-snow lookup") for n in (4877, 4902, 4905, 4934, 4937)],
    "hydraulic_heads_iterative": [(5241, 5241, "#9 layered-snow lookup"), (5245, 5245, "#9 layered-snow lookup")],
    "evapotranspiration": [(5179, 5191, "#9 frozen-soil ET")],
    "specific_fluxes": [
        (5480, 5484, "#9 snow conductivity"),
        (5536, 5551, "ArcVeg"),
        (5660, 5660, "clay layer disabled"),
        (5694, 5698, "#9 snow maxratio"),
        (5784, 5796, "ArcVeg and #9 snow interflow"),
        (5801, 5814, "drainage disabled"),
    ],
    "flux_restrictions_opt": [
        (6222, 6222, "#9 SE array"),
        (6226, 6230, "#9 negative snow offset"),
        (6556, 6556, "#9 SE array"),
    ],
    "groundwater_recharge": [(6685, 6701, "uncoupled/external groundwater options")],
    "calc_surface_interaction": [(5437, 5438, "tracers"), (5447, 5448, "tracers")],
}


# In-scope ranges the port does not reproduce (inclusive)
NOT_PORTED = {
    "pre_run": [(7732, 7913, "zonal travel-time registers and statistics (post_run phase, not ported)")],
    "post_run": [(13105, 13402, "travel-time zones, linear reservoirs, Qg (not ported)")],
    "run": [(11898, 12202, "zonal statistics and travel-time zone sums (not ported)")],
    "calc_surface_interaction": [(5391, 5452, "river reinfiltration from the river-link table "
                                               "(surface interaction 2, Doubs river cells; #8)")],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("doubs_objects")
    ap.add_argument("rhone_objects")
    ap.add_argument("--lines", action="store_true", help="list the executed lines that are not ported")
    args = ap.parse_args()
    data = {case: gcov_lines(args.source, obj, "unsatzon.cc")
            for case, obj in (("Doubs", args.doubs_objects), ("Rhone", args.rhone_objects))}
    print("| Function | In scope | Doubs | Rhone | Union executed | of which ported | executed, not ported |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    tot = [0, 0, 0, 0, 0]
    listing = []
    for name in FUNCTIONS:
        first, last = function_range(args.source, "unsatzon.cc", "unsatzonmodellklasse::" + name)
        excluded = {n for lo, hi, _ in EXCLUSIONS.get(name, []) for n in range(lo, hi + 1)}
        lines = [n for n, (count, _) in data["Doubs"].items()
                 if first <= n <= last and count is not None and n not in excluded]
        ran = {c: {n for n in lines if data[c][n][0] > 0} for c in data}
        union = ran["Doubs"] | ran["Rhone"]
        notp = {n: why for lo, hi, why in NOT_PORTED.get(name, []) for n in range(lo, hi + 1)}
        missed = sorted(n for n in union if n in notp)
        reasons = sorted({notp[n] for n in missed})
        row = [len(lines), len(ran["Doubs"]), len(ran["Rhone"]), len(union), len(union) - len(missed)]
        tot = [a + b for a, b in zip(tot, row, strict=True)]
        print(f"| {name} | " + " | ".join(str(x) for x in row) + f" | {len(missed)}"
              + (f": {'; '.join(reasons)}" if reasons else "") + " |")
        listing += [(name, n) for n in missed]
    print("| **total** | " + " | ".join(f"**{x}**" for x in tot) + f" | **{tot[3] - tot[4]}** |")
    if args.lines:
        for name, n in listing:
            print(f"{name} unsatzon.cc:{n}")


if __name__ == "__main__":
    main()
