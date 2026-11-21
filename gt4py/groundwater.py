"""Compare every active cell of groundwater before/after each soil substep.

Points: after_etreal -> after_gwflow. Geometry at start: zone and elevation.
Aquifer parameters (thickness, upper limit, k_x/y, storage, fixed h/q, cell sizes)
are read at each substep. The soil updates storage before the groundwater call.
Dynamic inputs: groundwater_distance, ponding_storage, GlacierizedCells (or
ice_firn), optional frozenSoilInGroundwater / SurfStorSiltingUp. Outputs:
heads/history, transmissivity, flow x/y, leakage, balance, baseflow, recharge,
river infiltration/exfiltration. The first gw_head_old_old_1 registration aliases
old_head and the hook selects it. True forelast/iteration history is not dumped.
No river table is needed: groundwater stores its address but never reads it.

The port is a DaCe program on the model cells (wasim_gt.cells, the cells of the zone grid): grids
are gathered to cells, the program runs, and its outputs are compared with the dump on the same
cells. warm_mean_ms is the time of the program call per substep (without gathering), from the
second substep on.

Usage: python gt4py/groundwater.py DUMP_DIR CONTROL_FILE --json result.json
CONTROL_FILE may be control.wasim or its resolved control.wasim.intern. The latter
is preferred. No case data or legacy parser belongs to wasim-gt.
"""
import argparse
import json
import pathlib
import re
from dataclasses import replace

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.processes.groundwater import Forcing, Parameters, State, groundwater, topology

OUTPUTS = ('gw_head_1', 'gw_head_old_1', 'gw_head_old_old_1',
           'gw_transmissivity_x_1', 'gw_transmissivity_y_1', 'gw_flow_x_1', 'gw_flow_y_1',
           'gw_balance1', 'baseflow', 'gw_leakage_1', 'groundwater_recharge',
           'river_infiltration', 'river_exfiltration')


class Records:
    """Memory-map the first matching registration of every dumped grid."""
    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        self.arrays = {}
        for line in (self.directory / 'manifest.txt').read_text().splitlines():
            point, name, rows, cols, levels = line.split()
            if int(levels):
                continue
            raw = np.memmap(self.directory / f'{point}__{name}.f64', dtype='f8', mode='r')
            self.arrays[point, name] = raw.reshape(-1, int(rows), int(cols))
        self.count = len(self.arrays['after_etreal', 'gw_head_1'])
        if len(self.arrays['after_gwflow', 'gw_head_1']) != self.count:
            raise ValueError('Before/after substep counts differ')

    def get(self, point, name, step=0):
        return self.arrays[point, name][step]


def section(text, name):
    body = text.split(f'[{name}]', 1)[1].split('\n[', 1)[0]
    return [line.split('#', 1)[0].strip() for line in body.splitlines()
            if line.split('#', 1)[0].strip()]


def read_parameters(control):
    path = pathlib.Path(control)
    if not path.name.endswith('.intern'):
        path = pathlib.Path(str(path) + '.intern')
    text = path.read_text(errors='replace')
    lines = section(text, 'groundwater_flow')
    method, maxiter, error, alpha, sor, layers = (float(lines[k]) for k in (2, 3, 4, 5, 6, 8))
    if method != 1 or layers != 1:
        raise ValueError('Only the single-aquifer Gauss-Seidel reference is supported')
    dt = float(lines[1]) * 60.0 / int(section(text, 'soil_surface_groundwater_substeps')[0])
    tuning = section(text, 'groundwater_tuning')
    match = re.search(r'frozenDepthScaling\s*=\s*([-+0-9.eE]+)', '\n'.join(tuning))
    return dict(dt=dt, alpha=alpha, sor=sor, max_iterations=int(maxiter), max_error=error,
                separate_zones=int(lines[0]) == 3, use_pond=int(lines[11]) == 1,
                frozen_depth_scaling=float(match[1]) if match else 1.0)


def compare_array(actual, expected, active, nodata=-9999.0):
    """Exactly compare finite values and explicitly compare exceptional-value patterns."""
    if actual.shape != expected.shape or active.shape != expected.shape:
        return dict(ok=False, reason='shape mismatch', max_abs=None, first=None)
    mismatch = np.zeros(actual.shape, dtype=bool)
    patterns = {}
    for name, predicate in [('nan', np.isnan), ('positive_infinity', np.isposinf),
                            ('negative_infinity', np.isneginf), ('nodata', lambda v: v == nodata)]:
        difference = active & (predicate(actual) != predicate(expected))
        patterns[name] = int(np.count_nonzero(difference))
        mismatch |= difference
    finite = active & np.isfinite(actual) & np.isfinite(expected) & (actual != nodata) & (expected != nodata)
    delta = np.zeros(actual.shape)
    delta[finite] = np.abs(actual[finite] - expected[finite])
    mismatch |= finite & (actual != expected)
    indices = np.argwhere(mismatch)
    first = None
    if len(indices):
        i, j = map(int, indices[0])
        first = dict(row=i + 1, column=j + 1, actual=float(actual[i, j]), expected=float(expected[i, j]))
    return dict(ok=not len(indices), max_abs=float(delta.max()), patterns=patterns, first=first)


def compare(directory, control, steps=None):
    dump = Records(directory)
    config = read_parameters(control)
    zone = dump.get('start', '[zone_grid]')
    layout = CellLayout.from_mask(np.isfinite(zone) & (zone != -9999.0))
    topo = topology(layout)
    cells = lambda array: layout.to_cells(array)
    static = lambda name: cells(dump.get('start', name))
    params = Parameters(*(static(n) for n in ('[zone_grid]', '[elevation_model]', 'aquifer_thickness_1',
        'aquifer_upper_limit1', 'gw_k_x_1', 'gw_k_y_1', 'gw_storage_coeff_1', 'gw_boundary_fix_h_1',
        'gw_boundary_fix_q_1', 'CellSizeX', 'CellSizeY')), **config)
    results = {name: dict(ok=True, max_abs=0.0, patterns={}, first=None) for name in OUTPUTS}
    times, iterations, oscillations, errors = [], [], [], []
    history = static("gw_head_1")
    iteration_history = history
    everywhere = np.ones(layout.n, dtype=bool)
    for step in range(min(steps or dump.count, dump.count)):
        before = lambda name, step=step: cells(dump.get('after_etreal', name, step))
        state = State(before('gw_head_1'), before('gw_head_old_1'), history, iteration_history,
                      *(before(n) for n in ('gw_flow_x_1', 'gw_flow_y_1', 'gw_balance1', 'baseflow')))
        ice_name = 'GlacierizedCells' if ('after_etreal', 'GlacierizedCells') in dump.arrays else 'ice_firn'
        ice = before(ice_name) if ('after_etreal', ice_name) in dump.arrays else np.zeros(layout.n)
        optional = lambda name, before=before: before(name) if ('after_etreal', name) in dump.arrays else None
        forcing = Forcing(*(before(n) for n in ('groundwater_distance', 'ponding_storage', 'gw_leakage_1',
            'groundwater_recharge', 'river_infiltration', 'river_exfiltration')), ice,
            optional('frozenSoilInGroundwater'), optional('SurfStorSiltingUp'))
        dynamic_params = replace(params, **{key: before(name) for key, name in
            [("thickness", "aquifer_thickness_1"), ("upper_limit", "aquifer_upper_limit1"),
             ("k_x", "gw_k_x_1"), ("k_y", "gw_k_y_1"), ("storage", "gw_storage_coeff_1"),
             ("fixed_head", "gw_boundary_fix_h_1"), ("fixed_flux", "gw_boundary_fix_q_1"),
             ("dx", "CellSizeX"), ("dy", "CellSizeY")]})
        result = groundwater(state, dynamic_params, forcing, topo)
        times.append(result.seconds)  # the program call only, without gathering and copies
        iterations.append(result.iterations)
        oscillations.append(result.oscillations)
        errors.append(result.max_error)
        st = result.state
        history, iteration_history = st.old_old_head, st.iteration_head
        fields = (st.head, st.old_head, st.old_head, result.transmissivity_x,
                  result.transmissivity_y, st.flow_x, st.flow_y, st.balance, st.baseflow, result.leakage,
                  result.recharge, result.river_infiltration, result.river_exfiltration)
        for name, output in zip(OUTPUTS, fields, strict=True):
            got = compare_array(output[:, None], cells(dump.get('after_gwflow', name, step))[:, None],
                                everywhere[:, None])
            if got['first'] is not None:  # cell index -> grid row and column (1-based, as WaSiM)
                c = got['first']['row'] - 1
                got['first'].update(row=int(layout.rows[c]) + 1, column=int(layout.cols[c]) + 1)
            summary = results[name]
            summary['ok'] &= got['ok']
            summary['max_abs'] = max(summary['max_abs'], got['max_abs'])
            for key, value in got['patterns'].items():
                summary['patterns'][key] = summary['patterns'].get(key, 0) + value
            if summary['first'] is None and got['first'] is not None:
                summary['first'] = dict(substep=step + 1, iterations=result.iterations, **got['first'])
        if any(not v['ok'] for v in results.values()):
            break
    return dict(ok=all(v['ok'] for v in results.values()), substeps=len(times),
                available_substeps=dump.count, active_cells=layout.n, parameters=config, grids=results,
                seconds_per_substep=times, iterations=iterations, oscillations=oscillations,
                final_errors=errors, warm_mean_ms=1000.0 * np.mean(times[1:]) if len(times) > 1 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump_dir')
    parser.add_argument('control_file')
    parser.add_argument('--json')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    result = compare(args.dump_dir, args.control_file, args.steps)
    text = json.dumps(result, indent=2)
    if args.json:
        pathlib.Path(args.json).write_text(text + '\n')
    print(text)
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
