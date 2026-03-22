import numpy as np

from ..network import Network


def connected_kva_allocation(net: Network, target_p_mw: float, power_factor: float = 0.9):
    '''
    Allocate load across an LFE network proportional to the connected kVA
    (transformer nameplate rating) feeding each load.

    For each load, finds the upstream transformer whose to_bus matches
    the load's bus_id and uses its mva_rating as the weight.
    Loads without a feeding transformer fall back to their existing |S|.

    Parameters
    ----------
    net : Network
        LFE internal network (must have loads and transformers populated)
    target_p_mw : float
        Total active power (MW) to distribute across all loads
    power_factor : float
        Power factor to derive Q from allocated P (default 0.9)
    '''
    if target_p_mw <= 0 or not net.loads:
        return

    # Build mapping: lv_bus -> total transformer mva_rating feeding it
    bus_to_trafo_mva = {}
    for xfmr in net.transformers.values():
        lv_bus = xfmr.to_bus
        bus_to_trafo_mva[lv_bus] = bus_to_trafo_mva.get(lv_bus, 0.0) + xfmr.mva_rating

    # For each load, get its connected capacity weight
    load_weights = {}
    for load_id, ld in net.loads.items():
        trafo_mva = bus_to_trafo_mva.get(ld.bus_id, 0.0)
        if trafo_mva > 0:
            load_weights[load_id] = trafo_mva
        else:
            # Fallback: use existing load |S|
            s = np.sqrt(ld.mw.sum()**2 + ld.mvar.sum()**2)
            load_weights[load_id] = s if s > 0 else 1e-6

    total_weight = sum(load_weights.values())
    if total_weight <= 0:
        return

    # Distribute target_p_mw proportionally and update load objects + bus aggregates
    q_factor = np.tan(np.arccos(power_factor))

    for load_id, weight in load_weights.items():
        ld = net.loads[load_id]
        allocated_p = (weight / total_weight) * target_p_mw
        allocated_q = allocated_p * q_factor

        # Determine number of active phases
        phases = []
        for p in range(3):
            if ld.mw[p] != 0 or ld.mvar[p] != 0:
                phases.append(p)
        if not phases:
            phases = [0, 1, 2]  # default to all phases

        n_phases = len(phases)
        p_per_phase = allocated_p / n_phases
        q_per_phase = allocated_q / n_phases

        # Remove old contribution from bus
        if ld.bus_id in net.buses:
            b = net.buses[ld.bus_id]
            for p in range(3):
                b.load_mw[p] -= ld.mw[p]
                b.load_mvar[p] -= ld.mvar[p]

        # Set new load values
        ld.mw = np.zeros(3)
        ld.mvar = np.zeros(3)
        for p in phases:
            ld.mw[p] = p_per_phase
            ld.mvar[p] = q_per_phase

        # Add new contribution to bus
        if ld.bus_id in net.buses:
            b = net.buses[ld.bus_id]
            for p in phases:
                b.load_mw[p] += p_per_phase
                b.load_mvar[p] += q_per_phase
