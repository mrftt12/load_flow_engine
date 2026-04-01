import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve

from load_flow_engine.network import Network
from load_flow_engine.solver import ThreePhaseLoadFlowSolver
from load_flow_engine.helpers import extract_res_bus, connected_kva_allocation2, _active_phases
from load_flow_engine.models import StudyCase
from load_flow_engine.enums import BusType


def create_study(circuit_key, max_iterations=50, solution_precision=1e-5, base_mva=10.0):
    sc  = StudyCase(max_iterations=max_iterations, solution_precision=solution_precision, base_mva=base_mva)
    net = Network(sc)
    return net

def run_pf(net, value, warm_start=False, warm_start_values=None, power_factor=0.9):
    """
    Runs power flow for the provided network after allocating load using the given value (target_p_mw).
    Optionally sets bus voltages from warm_start_values if warm_start is True.
    Args:
        net: Network object
        value: float, total active power (MW) to distribute across all loads
        warm_start: bool, if True, use warm_start_values for initial Vm_pu/Va_degree
        warm_start_values: dict, optional, mapping bus names to {'Vm_pu': float, 'Va_degree': float}
        power_factor: float, power factor for load allocation
    Returns:
        DataFrame of bus results with 'converged' column
    """
    # Allocate load before running power flow
    connected_kva_allocation2(net, value, power_factor=power_factor)

    # Set initial conditions for warm start if requested
    if warm_start and warm_start_values is not None:
        for bus in net.buses:
            # bus is a string (bus name), not an object with .name
            if bus in warm_start_values:
                vals = warm_start_values[bus]
                if 'Vm_pu' in vals:
                    net.buses[bus].Vm_pu = vals['Vm_pu']
                if 'Va_degree' in vals:
                    net.buses[bus].Va_degree = vals['Va_degree']

    solver = ThreePhaseLoadFlowSolver(net)
    solver.method = 'gs'
    converged = solver.solve()

    res_bus = extract_res_bus(net)
    res_bus["converged"] = converged
    net.res_bus = res_bus
    return res_bus


# Batch power flow function
def run_pf_batch(net, values_list, warm_start=False, power_factor=0.95):
    """
    Runs power flow for a sequence of values (floats) on the provided network.
    If warm_start is True, passes previous bus voltages as initial values for the next run.
    Args:
        net: Network object
        values_list: iterable of floats, each is total active power (MW) for one run
        warm_start: bool, if True, use previous results as initial values
        power_factor: float, power factor for load allocation
    Returns:
        List of results (one per value set)
    """
    results = []
    hr = 0
    prev_bus_voltages = None
    for value in values_list:
        hr += 1
        res = run_pf(
            net,
            value,
            warm_start=warm_start,
            warm_start_values=prev_bus_voltages,
            power_factor=power_factor
        )
        res = res.copy()
        res['hr'] = hr
        results.append(res)
        # Prepare voltages for next warm start
        if warm_start:
            prev_bus_voltages = {
                row['name']: {'Vm_pu': row['Vm_pu'], 'Va_degree': row['Va_degree']}
                for _, row in res.iterrows()
                if 'name' in row and 'Vm_pu' in row and 'Va_degree' in row
            }
    return results


def run_pf_batch_optimized(net, values_list, warm_start=False, power_factor=0.95):
    """
    Runs power flow for a sequence of values (floats) on the provided network.
    For values that are equal when rounded to one decimal, reuses the same result.
    If warm_start is True, passes previous bus voltages as initial values for the next run.
    Args:
        net: Network object
        values_list: iterable of floats, each is total active power (MW) for one run
        warm_start: bool, if True, use previous results as initial values
        power_factor: float, power factor for load allocation
    Returns:
        List of results (one per value set)
    """
    results = []
    prev_bus_voltages = None
    cache = {}
    hr = 0
    for value in values_list:
        hr += 1
        rounded = round(value, 1)
        if rounded in cache:
            res = cache[rounded]
        else:
            res = run_pf(
                net,
                value,
                warm_start=warm_start,
                warm_start_values=prev_bus_voltages,
                power_factor=power_factor
            )
            cache[rounded] = res
        res = res.copy()
        res['hr'] = hr
        results.append(res)
        if warm_start:
            prev_bus_voltages = {
                row['name']: {'Vm_pu': row['Vm_pu'], 'Va_degree': row['Va_degree']}
                for _, row in res.iterrows()
                if 'name' in row and 'Vm_pu' in row and 'Va_degree' in row
            }
    return results


# ---------------------------------------------------------------------------
# Fast batch solver – pre-computes topology-invariant quantities ONCE
# ---------------------------------------------------------------------------

def run_pf_batch_fast(net, values_list, warm_start=True, power_factor=0.95,
                      convergence_tol=1e-6):
    """
    High-performance time-series power flow.

    Runs 8760 (or any count) sequential power flows on an LFE Network
    where only the total MW changes each timestep.  All topology-dependent
    computations (Y-matrix build, BFS traversal, sparse LU factorisation,
    sub-matrix extraction) are performed **once** up front.  The inner loop
    is a tight Gauss-Seidel current-injection iteration that reuses the
    pre-factored LU object.

    Speedup sources vs run_pf_batch / run_pf_batch_optimized
    ---------------------------------------------------------
    1.  splu(Y_ff) factored once instead of per-timestep  (~60-70 % of time)
    2.  BFS / adjacency / index bookkeeping done once
    3.  Load-weight vector pre-computed; allocation is a scalar x vector op
    4.  No ThreePhaseLoadFlowSolver instantiation per step
    5.  No DataFrame construction per step; results accumulated in arrays
    6.  Warm start reuses voltage vector in-place (no dict round-trip)
    7.  Single pd.DataFrame built at the end from pre-allocated arrays

    Parameters
    ----------
    net : Network
        Built LFE network (will be mutated - load values and bus voltages
        are overwritten).
    values_list : iterable of float
        Total active power (MW) for each timestep.
    warm_start : bool
        If True, reuse the previous solution as the initial voltage guess.
    power_factor : float
        Power factor used to derive Q from allocated P.
    convergence_tol : float
        Maximum voltage-change tolerance for the GS inner loop.

    Returns
    -------
    pd.DataFrame
        Concatenated bus results for all timesteps with an ``hr`` column.
    """
    values_arr = np.asarray(list(values_list), dtype=float)
    n_steps = len(values_arr)

    # ------------------------------------------------------------------
    # 1. Ensure Y_abc is built
    # ------------------------------------------------------------------
    if net.Y_abc is None:
        net.build()

    buses = net._bus_order
    N = len(buses)
    base_mva = net.study_case.base_mva
    max_iter = net.study_case.max_iterations
    sol_prec = net.study_case.solution_precision

    # ------------------------------------------------------------------
    # 2. Pre-compute load-weight vector (same logic as allocation2)
    # ------------------------------------------------------------------
    bus_to_trafo_mva = {}
    for xfmr in net.transformers.values():
        lv = xfmr.to_bus
        bus_to_trafo_mva[lv] = bus_to_trafo_mva.get(lv, 0.0) + xfmr.mva_rating

    load_ids = list(net.loads.keys())
    load_weights_raw = {}
    load_phases_map = {}
    for lid in load_ids:
        ld = net.loads[lid]
        trafo_mva = bus_to_trafo_mva.get(ld.bus_id, 0.0)
        if trafo_mva > 0:
            load_weights_raw[lid] = trafo_mva
        else:
            s = np.sqrt(ld.mw.sum()**2 + ld.mvar.sum()**2)
            load_weights_raw[lid] = s if s > 0 else 1e-6
        phases = [p for p in range(3) if ld.mw[p] != 0 or ld.mvar[p] != 0]
        if not phases:
            phases = [0, 1, 2]
        load_phases_map[lid] = phases

    total_weight = sum(load_weights_raw.values())

    q_factor = np.tan(np.arccos(power_factor))
    inv_base = 1.0 / base_mva

    # Build a *delta S_sch* vector: for target_p_mw=1.0 MW, what is the
    # per-unit S contribution at each bus-phase node?
    S_per_mw = np.zeros(3 * N, dtype=complex)
    for lid in load_ids:
        ld = net.loads[lid]
        w = load_weights_raw[lid]
        frac = w / total_weight if total_weight > 0 else 0.0
        phases = load_phases_map[lid]
        n_ph = len(phases)
        p_per_phase = frac / n_ph
        q_per_phase = p_per_phase * q_factor
        if ld.bus_id in net._bus_index:
            bi = net._bus_index[ld.bus_id]
            for p in phases:
                S_per_mw[3 * bi + p] -= complex(p_per_phase, q_per_phase) * inv_base

    # Capture generation (constant across timesteps)
    S_gen = np.zeros(3 * N, dtype=complex)
    for i, bid in enumerate(buses):
        b = net.buses[bid]
        for p in range(3):
            S_gen[3 * i + p] = complex(b.gen_mw[p], b.gen_mvar[p]) * inv_base

    # ------------------------------------------------------------------
    # 3. Pre-compute topology: BFS, free/slack indices, LU factorisation
    # ------------------------------------------------------------------
    slack_set = set()
    for i, bid in enumerate(buses):
        if net.buses[bid].bus_type == BusType.SLACK:
            slack_set.update([3 * i, 3 * i + 1, 3 * i + 2])

    adj = {k: [] for k in range(3 * N)}
    for br in net.branches.values():
        fi = net._bus_index[br.from_bus]
        ti = net._bus_index[br.to_bus]
        mask = _active_phases(br.phase_type)
        for p in mask:
            a, b_node = 3 * fi + p, 3 * ti + p
            adj[a].append(b_node)
            adj[b_node].append(a)
    for xf in net.transformers.values():
        fi = net._bus_index[xf.from_bus]
        ti = net._bus_index[xf.to_bus]
        mask = _active_phases(xf.phase_type)
        for p in mask:
            a, b_node = 3 * fi + p, 3 * ti + p
            adj[a].append(b_node)
            adj[b_node].append(a)

    reachable = set()
    queue = list(slack_set)
    reachable.update(queue)
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if nb not in reachable:
                reachable.add(nb)
                queue.append(nb)

    Y_csc = csc_matrix(net.Y_abc)
    Y_diag = np.array(Y_csc.diagonal()).flatten()
    free_idx = np.array(sorted(k for k in reachable
                                if k not in slack_set
                                and abs(Y_diag[k]) > 1e-10), dtype=int)
    slack_idx = np.array(sorted(slack_set), dtype=int)
    n_free = len(free_idx)

    Y_ff = Y_csc[free_idx][:, free_idx]
    Y_fs = Y_csc[free_idx][:, slack_idx]

    # LU factorisation - the expensive step, done ONCE
    lu = None
    try:
        lu = splu(Y_ff)
    except Exception:
        warnings.warn("LU factorisation of Y_ff failed; falling back to spsolve per step.")

    # ------------------------------------------------------------------
    # 4. Initial voltage vector (flat start)
    # ------------------------------------------------------------------
    nom_ang = np.deg2rad([0.0, -120.0, 120.0])
    V = np.zeros(3 * N, dtype=complex)
    if net.study_case.flat_start:
        for i in range(N):
            for p in range(3):
                V[3 * i + p] = 1.0 * np.exp(1j * nom_ang[p])
    else:
        for i, bid in enumerate(buses):
            b = net.buses[bid]
            for p in range(3):
                V[3 * i + p] = b.ini_v_mag[p] * np.exp(1j * np.deg2rad(b.ini_v_ang[p]))

    V_slack = V[slack_idx].copy()
    rhs_slack = Y_fs @ V_slack  # constant

    # ------------------------------------------------------------------
    # 5. Pre-compute Fortescue inverse for result extraction
    # ------------------------------------------------------------------
    a_fort = np.exp(1j * 2 * np.pi / 3)
    A_inv = (1.0 / 3.0) * np.array([
        [1, 1, 1],
        [1, a_fort, a_fort**2],
        [1, a_fort**2, a_fort]
    ])

    # Pre-allocate result arrays
    n_buses = N
    all_vm_pu = np.empty((n_steps, n_buses), dtype=float)
    all_va_deg = np.empty((n_steps, n_buses), dtype=float)
    all_v_abc_mag = np.empty((n_steps, n_buses, 3), dtype=float)
    all_v_abc_ang = np.empty((n_steps, n_buses, 3), dtype=float)
    all_converged = np.empty(n_steps, dtype=bool)

    bus_names = []
    bus_ids_out = []
    for bid in buses:
        b = net.buses[bid]
        bus_names.append(b.name if hasattr(b, 'name') else '')
        try:
            bus_ids_out.append(int(bid))
        except ValueError:
            bus_ids_out.append(bid)

    # ------------------------------------------------------------------
    # 6. Main time-series loop - tight GS iteration
    # ------------------------------------------------------------------
    S_per_mw_free = S_per_mw[free_idx]
    S_gen_free = S_gen[free_idx]

    for step in range(n_steps):
        target_p = values_arr[step]

        # S_sch for free nodes = generation + load_scaling * target_p
        S_sch_free = S_gen_free + S_per_mw_free * target_p

        if not warm_start:
            # Reset to flat start each timestep
            if net.study_case.flat_start:
                for i in range(N):
                    for p in range(3):
                        V[3 * i + p] = 1.0 * np.exp(1j * nom_ang[p])

        converged = False
        for iteration in range(max_iter):
            I_inj_free = np.conj(S_sch_free) / np.conj(V[free_idx])
            rhs = I_inj_free - rhs_slack

            if lu is not None:
                V_free_new = lu.solve(rhs)
            else:
                V_free_new = spsolve(Y_ff, rhs)

            max_dv = np.max(np.abs(V_free_new - V[free_idx]))
            V[free_idx] = V_free_new

            if max_dv < convergence_tol:
                I_calc = np.array(Y_csc @ V).flatten()
                S_calc = V * np.conj(I_calc)
                max_mis = np.max(np.abs(
                    S_sch_free - S_calc[free_idx])) * base_mva
                if max_mis < sol_prec:
                    converged = True
                    break

        all_converged[step] = converged

        # Extract bus results directly from V (no DataFrame per step)
        for i in range(n_buses):
            v3 = V[3 * i: 3 * i + 3]
            mag = np.abs(v3)
            ang = np.rad2deg(np.angle(v3))
            all_v_abc_mag[step, i, :] = mag
            all_v_abc_ang[step, i, :] = ang
            V_012 = A_inv @ v3
            all_vm_pu[step, i] = abs(V_012[1])
            all_va_deg[step, i] = np.rad2deg(np.angle(V_012[1]))

    # ------------------------------------------------------------------
    # 7. Build a single DataFrame from accumulated arrays
    # ------------------------------------------------------------------
    rows_per_step = n_buses
    hr_col = np.repeat(np.arange(1, n_steps + 1), rows_per_step)
    name_col = np.tile(bus_names, n_steps)
    conv_col = np.repeat(all_converged, rows_per_step)

    result_df = pd.DataFrame({
        'bus': np.tile(bus_ids_out, n_steps),
        'name': name_col,
        'vm_pu': all_vm_pu.ravel(),
        'va_degree': all_va_deg.ravel(),
        'v_a_pu': all_v_abc_mag[:, :, 0].ravel(),
        'v_b_pu': all_v_abc_mag[:, :, 1].ravel(),
        'v_c_pu': all_v_abc_mag[:, :, 2].ravel(),
        'va_a_degree': all_v_abc_ang[:, :, 0].ravel(),
        'va_b_degree': all_v_abc_ang[:, :, 1].ravel(),
        'va_c_degree': all_v_abc_ang[:, :, 2].ravel(),
        'converged': conv_col,
        'hr': hr_col,
    })
    result_df.set_index('bus', inplace=True)

    # Update network bus objects with final timestep solution
    for i, bid in enumerate(buses):
        b = net.buses[bid]
        b.v_mag[:] = all_v_abc_mag[-1, i, :]
        b.v_ang[:] = all_v_abc_ang[-1, i, :]

    return result_df