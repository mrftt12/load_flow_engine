import numpy as np
import pandas as pd

from ..network import Network


def get_loading_by_phase(net: Network) -> dict:
    '''
    Calculates total real and reactive power per phase (A, B, C)
    for loads and generators in an LFE network.

    Returns a dict with keys:
        P_A_MW, P_B_MW, P_C_MW, P_MW_TOTAL,
        Q_A_MVAR, Q_B_MVAR, Q_C_MVAR, Q_MVAR_TOTAL,
        PGEN_A_MW, PGEN_B_MW, PGEN_C_MW, PGEN_MW_TOTAL
    '''
    p_load = np.zeros(3)
    q_load = np.zeros(3)
    p_gen = np.zeros(3)

    for ld in net.loads.values():
        p_load += ld.mw
        q_load += ld.mvar

    for g in net.generators.values():
        p_gen += g.mw

    return {
        'P_A_MW': p_load[0],
        'P_B_MW': p_load[1],
        'P_C_MW': p_load[2],
        'P_MW_TOTAL': p_load.sum(),
        'Q_A_MVAR': q_load[0],
        'Q_B_MVAR': q_load[1],
        'Q_C_MVAR': q_load[2],
        'Q_MVAR_TOTAL': q_load.sum(),
        'PGEN_A_MW': p_gen[0],
        'PGEN_B_MW': p_gen[1],
        'PGEN_C_MW': p_gen[2],
        'PGEN_MW_TOTAL': p_gen.sum(),
    }


def extract_res_bus(net: Network) -> pd.DataFrame:
    '''
    Extracts the solved bus voltages from the internal ETAP Network model
    into a pandas DataFrame, similar to pandapower's res_bus.
    '''
    data = []

    # Transformation matrix for sequence components
    a = np.exp(1j * 2 * np.pi / 3)
    A_inv = (1/3) * np.array([
        [1, 1, 1],
        [1, a, a**2],
        [1, a**2, a]
    ])

    for bus_id, bus in net.buses.items():
        # Reconstruct complex phase voltages
        V_abc = bus.v_mag * np.exp(1j * np.deg2rad(bus.v_ang))

        # Calculate sequence components to get positive sequence for standard fields
        V_012 = A_inv @ V_abc
        vm_pu = abs(V_012[1]) # Positive sequence voltage magnitude
        va_degree = np.rad2deg(np.angle(V_012[1])) # Positive sequence voltage angle

        row = {
            'name': bus.name if hasattr(bus, 'name') else '',
            'vm_pu': vm_pu,
            'va_degree': va_degree,
            'v_a_pu': bus.v_mag[0],
            'v_b_pu': bus.v_mag[1],
            'v_c_pu': bus.v_mag[2],
            'va_a_degree': bus.v_ang[0],
            'va_b_degree': bus.v_ang[1],
            'va_c_degree': bus.v_ang[2]
        }

        try:
            row['bus'] = int(bus_id)
        except ValueError:
            row['bus'] = bus_id

        data.append(row)

    res_bus = pd.DataFrame(data)
    res_bus.set_index('bus', inplace=True)
    return res_bus
