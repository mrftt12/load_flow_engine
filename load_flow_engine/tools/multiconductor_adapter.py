import warnings

import numpy as np
import pandas as pd
import pandapower as pp

warnings.filterwarnings("ignore", category=UserWarning, module='pandapower')
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ..network import Network
from ..models import Bus, Branch, Transformer, Load, Generator, Shunt, Switch, StudyCase
from ..enums import BusType, PhaseType

def build_internal_from_pandapower(pp_net: pp.pandapowerNet, base_mva: float = 100.0) -> Network:
    sc = StudyCase(max_iterations=50, solution_precision=1e-4, base_mva=base_mva)
    net = Network(sc)
    
    bus_vk = {}
    
    # 1. Process buses
    for idx, row in pp_net.bus.iterrows():
        b_id = idx[0] if isinstance(idx, tuple) else idx
        b_id = str(int(b_id))
        
        if b_id in bus_vk:
            continue
            
        b_type = BusType.PQ
        if 'ext_grid' in pp_net and not pp_net.ext_grid.empty and int(b_id) in pp_net.ext_grid.bus.values:
            b_type = BusType.SLACK
        elif 'gen' in pp_net and not pp_net.gen.empty and int(b_id) in pp_net.gen.bus.values:
            b_type = BusType.PV
        
        vk_kv = float(row.vn_kv)
        bus_vk[b_id] = vk_kv
        bus_name = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
        net.add_bus(Bus(b_id, b_type, PhaseType.ABC, base_kv=vk_kv, name=bus_name))
        
    # 2. Process external grids
    if 'ext_grid' in pp_net and not pp_net.ext_grid.empty:
        for idx, row in pp_net.ext_grid.iterrows():
            bus_id = str(int(row.bus))
            vm_pu = float(row.vm_pu)
            va_degree = float(row.va_degree) if 'va_degree' in row and not np.isnan(row.va_degree) else 0.0
            
            b = net.buses[bus_id]
            b.bus_type = BusType.SLACK
            b.ini_v_mag = np.full(3, vm_pu)
            b.ini_v_ang = np.array([va_degree, va_degree - 120, va_degree + 120])
            b.v_mag = b.ini_v_mag.copy()
            b.v_ang = b.ini_v_ang.copy()

    # 3. Process lines — group by line_id, determine actual phases
    if 'line' in pp_net and not pp_net.line.empty:
        line_df = pp_net.line
        is_mi = isinstance(line_df.index, pd.MultiIndex)
        
        # Group conductors by line_id
        if is_mi:
            idx_name = line_df.index.names[0]
            line_ids = line_df.index.get_level_values(idx_name).unique()
        else:
            line_ids = line_df.index.unique()
        
        phase_map = {
            (0,): PhaseType.A, (1,): PhaseType.B, (2,): PhaseType.C,
            (0,1): PhaseType.AB, (1,2): PhaseType.BC, (0,2): PhaseType.CA,
            (0,1,2): PhaseType.ABC,
        }
        
        for l_id in line_ids:
            if is_mi:
                sub = line_df.loc[l_id]
                if not isinstance(sub, pd.DataFrame):
                    sub = line_df.loc[[l_id]]
            else:
                sub = line_df.loc[[l_id]]
            
            first_row = sub.iloc[0]
            f_bus = str(int(first_row.from_bus))
            t_bus = str(int(first_row.to_bus))
            length_km = float(first_row.length_km)
            f_vk_kv = bus_vk.get(f_bus, 12.47)
            Zbase = f_vk_kv**2 / base_mva
            
            # Determine power phases (1=A, 2=B, 3=C; 0=neutral, skip)
            if 'from_phase' in sub.columns:
                raw_phases = sorted(sub['from_phase'].unique())
                power_phases = sorted([int(p) - 1 for p in raw_phases if int(p) >= 1 and int(p) <= 3])
            else:
                power_phases = [0, 1, 2]
            
            if not power_phases:
                continue  # neutral-only line, skip
            
            pt = phase_map.get(tuple(power_phases), PhaseType.ABC)
            n_power = len(power_phases)
            
            r_ohm, x_ohm, r0_ohm, x0_ohm = 0.0, 0.0, 0.0, 0.0
            b1_mho = 0.0
            ampacity = 9999.0
            
            if hasattr(pp_net, 'std_types') and 'std_type' in first_row and str(first_row.std_type) != 'nan':
                st_type = first_row.std_type
                cat = first_row.model_type if 'model_type' in first_row else 'sequence'
                if cat in pp_net.std_types and st_type in pp_net.std_types[cat]:
                    std_dict = pp_net.std_types[cat][st_type]
                    if cat == 'sequence':
                        r_ohm = float(std_dict.get('r_ohm_per_km', 0.0))
                        x_ohm = float(std_dict.get('x_ohm_per_km', 0.0))
                        r0_ohm = float(std_dict.get('r0_ohm_per_km', r_ohm * 3))
                        x0_ohm = float(std_dict.get('x0_ohm_per_km', x_ohm * 3))
                        c_nf = float(std_dict.get('c_nf_per_km', 0.0))
                        b1_mho = 2 * np.pi * 60 * (c_nf * 1e-9)
                        ampacity = float(std_dict.get('max_i_ka', 9999.0)) * 1000
                    elif cat == 'matrix':
                        # Determine actual conductor count from std_type
                        n_cond_total = len(raw_phases) if 'from_phase' in sub.columns else 3
                        
                        # Build full n_cond x n_cond Z matrix from matrix std_type
                        R_mat = np.zeros((n_cond_total, n_cond_total))
                        X_mat = np.zeros((n_cond_total, n_cond_total))
                        B_mat = np.zeros((n_cond_total, n_cond_total))
                        for col_i in range(n_cond_total):
                            r_col = std_dict.get(f'r_{col_i+1}_ohm_per_km')
                            x_col = std_dict.get(f'x_{col_i+1}_ohm_per_km')
                            b_col = std_dict.get(f'b_{col_i+1}_us_per_km')
                            if r_col is not None:
                                for row_i in range(min(len(r_col), n_cond_total)):
                                    R_mat[row_i, col_i] = float(r_col[row_i])
                            if x_col is not None:
                                for row_i in range(min(len(x_col), n_cond_total)):
                                    X_mat[row_i, col_i] = float(x_col[row_i])
                            if b_col is not None:
                                for row_i in range(min(len(b_col), n_cond_total)):
                                    B_mat[row_i, col_i] = float(b_col[row_i]) * 1e-6
                        
                        Z_full = R_mat + 1j * X_mat
                        
                        # Extract power-phase sub-matrix (exclude neutral = phase 0)
                        # Conductor ordering matches from_phase ordering in the multi-index
                        if 'from_phase' in sub.columns:
                            sorted_raw = sorted(raw_phases)
                            power_cond_idx = [sorted_raw.index(p) for p in sorted_raw if int(p) >= 1]
                        else:
                            power_cond_idx = list(range(min(n_cond_total, 3)))
                        
                        Z_power = Z_full[np.ix_(power_cond_idx, power_cond_idx)]
                        B_power = B_mat[np.ix_(power_cond_idx, power_cond_idx)]
                        
                        # Check if impedances are placeholders (diagonal < 1e-3)
                        is_placeholder = all(abs(Z_power[d, d]) < 1e-3 for d in range(len(power_cond_idx)))
                        
                        if is_placeholder:
                            # Use reasonable default impedances for distribution lines
                            # Typical 1/0 ACSR: R1≈0.37, X1≈0.48, R0≈0.55, X0≈1.5 ohm/km
                            r_ohm = 0.37
                            x_ohm = 0.48
                            r0_ohm = 0.55
                            x0_ohm = 1.5
                        elif n_power == 3:
                            # Fortescue transform: Z012 = A^-1 Z_abc A
                            a_op = np.exp(1j * 2 * np.pi / 3)
                            A_f = np.array([[1, 1, 1], [1, a_op**2, a_op], [1, a_op, a_op**2]])
                            A_inv = np.linalg.inv(A_f)
                            Z_012 = A_inv @ Z_power @ A_f
                            z1 = Z_012[1, 1]  # positive sequence
                            z0 = Z_012[0, 0]  # zero sequence
                            r_ohm = max(z1.real, 0.001)
                            x_ohm = max(z1.imag, 0.001)
                            r0_ohm = max(z0.real, 0.001)
                            x0_ohm = max(z0.imag, 0.001)
                        elif n_power == 2:
                            # Average of diagonal self-impedances
                            z_avg = (Z_power[0, 0] + Z_power[1, 1]) / 2
                            r_ohm = max(z_avg.real, 0.001)
                            x_ohm = max(z_avg.imag, 0.001)
                            r0_ohm = r_ohm * 3
                            x0_ohm = x_ohm * 3
                        else:
                            # Single phase
                            r_ohm = max(Z_power[0, 0].real, 0.001)
                            x_ohm = max(Z_power[0, 0].imag, 0.001)
                            r0_ohm = r_ohm * 3
                            x0_ohm = x_ohm * 3
                        
                        b1_mho = np.mean(np.diag(B_power)) if B_power.size > 0 else 0.0
                        
                        amp = std_dict.get('max_i_ka', 9999.0)
                        ampacity = float(amp[0]) if isinstance(amp, list) else float(amp)
                        ampacity *= 1000
            elif 'r_ohm_per_km' in first_row:
                r_ohm = float(first_row.r_ohm_per_km)
                x_ohm = float(first_row.x_ohm_per_km)
                r0_ohm = float(first_row.get('r0_ohm_per_km', r_ohm * 3))
                x0_ohm = float(first_row.get('x0_ohm_per_km', x_ohm * 3))
                c_nf = float(first_row.get('c_nf_per_km', 0.0))
                b1_mho = 2 * np.pi * 60 * (c_nf * 1e-9)
                ampacity = float(first_row.get('max_i_ka', 9999.0)) * 1000
                
            r1 = (r_ohm * length_km) / Zbase if Zbase > 0 else 0.0
            x1 = (x_ohm * length_km) / Zbase if Zbase > 0 else 0.0
            r0 = (r0_ohm * length_km) / Zbase if Zbase > 0 else 0.0
            x0 = (x0_ohm * length_km) / Zbase if Zbase > 0 else 0.0
            b1_pu = b1_mho * Zbase
            
            line_name = str(first_row['name']) if 'name' in first_row and str(first_row['name']) != 'nan' else ''
            net.add_branch(Branch(
                id=f"Line_{l_id}", name=line_name,
                from_bus=f_bus, to_bus=t_bus, phase_type=pt,
                r1=max(r1, 1e-6), x1=max(x1, 1e-6), r0=max(r0, 1e-6), x0=max(x0, 1e-6), b1=b1_pu,
                ampacity=np.full(3, ampacity)
            ))
            
    # 4. Process transformers
    if 'trafo' in pp_net and not pp_net.trafo.empty:
        for idx, row in pp_net.trafo.iterrows():
            hv_bus = str(int(row.hv_bus))
            lv_bus = str(int(row.lv_bus))
            sn_mva, vk_pct, vkr_pct = 1.0, 1.0, 1.0
            conn_hv, conn_lv = 'wye_grounded', 'wye_grounded'
            
            if hasattr(pp_net, 'std_types') and 'trafo' in pp_net.std_types and 'std_type' in row and str(row.std_type) != 'nan' and row.std_type in pp_net.std_types['trafo']:
                std_dict = pp_net.std_types['trafo'][row.std_type]
                sn_mva = float(std_dict.get('sn_mva', 1.0))
                vk_pct = float(std_dict.get('vk_percent', 1.0))
                vkr_pct = float(std_dict.get('vkr_percent', 1.0))
                vg = std_dict.get('vector_group', 'Yy0')
                conn_hv = 'wye_grounded' if 'Y' in vg else 'delta'
                conn_lv = 'wye_grounded' if 'y' in vg else 'delta'
            elif 'sn_mva' in row:
                sn_mva = float(row.sn_mva)
                vk_pct = float(row.vk_percent)
                vkr_pct = float(row.vkr_percent)
                vg = row.get('vector_group', 'Yy0')
                conn_hv = 'wye_grounded' if 'Y' in vg else 'delta'
                conn_lv = 'wye_grounded' if 'y' in vg else 'delta'
                
            r1 = vkr_pct / 100.0
            z1 = vk_pct / 100.0
            x1 = np.sqrt(max(z1**2 - r1**2, 0))
            
            t_id = idx[0] if isinstance(idx, tuple) else idx
            trafo_name = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
            net.add_transformer(Transformer(
                id=f"Trafo_{t_id}", name=trafo_name,
                from_bus=hv_bus, to_bus=lv_bus, phase_type=PhaseType.ABC,
                r1=r1, x1=x1, r0=r1, x0=x1,
                mva_rating=sn_mva,
                conn_primary=conn_hv, conn_secondary=conn_lv
            ))
            
    # 5. Process symmetric loads
    if 'load' in pp_net and not pp_net.load.empty:
        for idx, row in pp_net.load.iterrows():
            b_id = str(int(row.bus))
            p_ph = float(row.p_mw) / 3.0
            q_ph = float(row.q_mvar) / 3.0
            ld_id = idx[0] if isinstance(idx, tuple) else idx
            load_name = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
            net.add_load(Load(
                id=f"Load_{ld_id}", name=load_name,
                bus_id=b_id, phase_type=PhaseType.ABC,
                mw=np.full(3, p_ph), mvar=np.full(3, q_ph)
            ))
            
    # 6. Process asymmetric loads (grouped by load index, phase from from_phase)
    if 'asymmetric_load' in pp_net and not pp_net.asymmetric_load.empty:
        al = pp_net.asymmetric_load
        has_from_phase = 'from_phase' in al.columns
        if has_from_phase and isinstance(al.index, pd.MultiIndex):
            idx_name = al.index.names[0]
            for ld_id in al.index.get_level_values(idx_name).unique():
                sub = al.loc[ld_id] if isinstance(al.loc[ld_id], pd.DataFrame) else al.loc[[ld_id]]
                b_id = str(int(sub['bus'].iloc[0]))
                p_mw = np.zeros(3)
                q_mvar = np.zeros(3)
                active_phases = []
                for _, row in sub.iterrows():
                    phase = int(row['from_phase']) - 1  # 1->A(0), 2->B(1), 3->C(2)
                    if 0 <= phase <= 2:
                        p_mw[phase] += float(row.get('p_mw', 0) or 0)
                        q_mvar[phase] += float(row.get('q_mvar', 0) or 0)
                        if phase not in active_phases:
                            active_phases.append(phase)
                # Determine PhaseType from active phases
                phase_map = {
                    (0,): PhaseType.A, (1,): PhaseType.B, (2,): PhaseType.C,
                    (0,1): PhaseType.AB, (1,2): PhaseType.BC, (0,2): PhaseType.CA,
                    (0,1,2): PhaseType.ABC,
                }
                pt = phase_map.get(tuple(sorted(active_phases)), PhaseType.ABC)
                aload_name = str(sub['name'].iloc[0]) if 'name' in sub.columns and str(sub['name'].iloc[0]) != 'nan' else ''
                net.add_load(Load(
                    id=f"AsymLoad_{ld_id}", name=aload_name,
                    bus_id=b_id, phase_type=pt,
                    mw=p_mw, mvar=q_mvar
                ))
        else:
            for idx, row in al.iterrows():
                if 'bus' in row and not np.isnan(row.bus):
                    b_id = str(int(row.bus))
                    p_mw = np.zeros(3)
                    q_mvar = np.zeros(3)
                    if has_from_phase and not np.isnan(row['from_phase']):
                        phase = int(row['from_phase']) - 1
                        if 0 <= phase <= 2:
                            p_mw[phase] = float(row.get('p_mw', 0) or 0)
                            q_mvar[phase] = float(row.get('q_mvar', 0) or 0)
                    else:
                        if 'p_a_mw' in row and not np.isnan(row.p_a_mw): p_mw[0] = float(row.p_a_mw)
                        if 'p_b_mw' in row and not np.isnan(row.p_b_mw): p_mw[1] = float(row.p_b_mw)
                        if 'p_c_mw' in row and not np.isnan(row.p_c_mw): p_mw[2] = float(row.p_c_mw)
                        if 'q_a_mvar' in row and not np.isnan(row.q_a_mvar): q_mvar[0] = float(row.q_a_mvar)
                        if 'q_b_mvar' in row and not np.isnan(row.q_b_mvar): q_mvar[1] = float(row.q_b_mvar)
                        if 'q_c_mvar' in row and not np.isnan(row.q_c_mvar): q_mvar[2] = float(row.q_c_mvar)
                    ld_id = idx[0] if isinstance(idx, tuple) else idx
                    aload_name2 = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
                    net.add_load(Load(
                        id=f"AsymLoad_{ld_id}", name=aload_name2,
                        bus_id=b_id, phase_type=PhaseType.ABC,
                        mw=p_mw, mvar=q_mvar
                    ))

    # 7. Process generators (sgen — symmetric)
    if 'sgen' in pp_net and not pp_net.sgen.empty:
        for idx, row in pp_net.sgen.iterrows():
            b_id = str(int(row.bus))
            p_ph = float(row.p_mw) / 3.0
            g_id = idx[0] if isinstance(idx, tuple) else idx
            sgen_name = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
            net.add_generator(Generator(
                id=f"SGen_{g_id}", name=sgen_name,
                bus_id=b_id, bus_type=BusType.PQ,
                mw=np.full(3, p_ph)
            ))

    # 8. Process asymmetric generators (asymmetric_sgen)
    if 'asymmetric_sgen' in pp_net and not pp_net.asymmetric_sgen.empty:
        asgen = pp_net.asymmetric_sgen
        if isinstance(asgen.index, pd.MultiIndex):
            group_level = asgen.index.names[0]
            for g_id in asgen.index.get_level_values(group_level).unique():
                sub = asgen.loc[g_id] if isinstance(asgen.loc[g_id], pd.DataFrame) else asgen.loc[[g_id]]
                b_id = str(int(sub['bus'].iloc[0]))
                p_mw = np.zeros(3)
                q_mvar = np.zeros(3)
                for _, row in sub.iterrows():
                    phase = int(row['from_phase']) - 1 if 'from_phase' in row and not np.isnan(row['from_phase']) else 0
                    if 0 <= phase <= 2:
                        p_mw[phase] += float(row.get('p_mw', 0) or 0)
                        q_mvar[phase] += float(row.get('q_mvar', 0) or 0)
                asgen_name = str(sub['name'].iloc[0]) if 'name' in sub.columns and str(sub['name'].iloc[0]) != 'nan' else ''
                net.add_generator(Generator(
                    id=f"AsymSGen_{g_id}", name=asgen_name,
                    bus_id=b_id, bus_type=BusType.PQ,
                    mw=p_mw
                ))
        else:
            for idx, row in asgen.iterrows():
                b_id = str(int(row.bus))
                p_mw = np.zeros(3)
                phase = int(row['from_phase']) - 1 if 'from_phase' in row and not np.isnan(row['from_phase']) else 0
                if 0 <= phase <= 2:
                    p_mw[phase] = float(row.get('p_mw', 0) or 0)
                g_id = idx[0] if isinstance(idx, tuple) else idx
                asgen_name2 = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
                net.add_generator(Generator(
                    id=f"AsymSGen_{g_id}", name=asgen_name2,
                    bus_id=b_id, bus_type=BusType.PQ,
                    mw=p_mw
                ))

    # 9. Process single-phase transformers (trafo1ph)
    if 'trafo1ph' in pp_net and not pp_net.trafo1ph.empty:
        trafo1ph = pp_net.trafo1ph
        idx_name = trafo1ph.index.names[0] if isinstance(trafo1ph.index, pd.MultiIndex) else None
        if idx_name:
            for t_id in trafo1ph.index.get_level_values(idx_name).unique():
                sub = trafo1ph.loc[t_id] if isinstance(trafo1ph.loc[t_id], pd.DataFrame) else trafo1ph.loc[[t_id]]
                buses_in_trafo = sub.index.get_level_values('bus').unique() if 'bus' in sub.index.names else []
                if len(buses_in_trafo) < 2:
                    continue
                vn_kvs = sub['vn_kv'].values
                hv_idx = np.argmax(vn_kvs)
                lv_idx = np.argmin(vn_kvs)
                if isinstance(sub.index, pd.MultiIndex) and 'bus' in sub.index.names:
                    hv_bus_vals = sub.index.get_level_values('bus').unique()
                    # Group by bus to find HV and LV
                    bus_vn = {}
                    for b in hv_bus_vals:
                        b_sub = sub.xs(b, level='bus') if 'bus' in sub.index.names else sub
                        bus_vn[b] = b_sub['vn_kv'].iloc[0]
                    sorted_buses = sorted(bus_vn.items(), key=lambda x: x[1], reverse=True)
                    hv_bus = str(int(sorted_buses[0][0]))
                    lv_bus = str(int(sorted_buses[1][0]))
                else:
                    continue
                sn_mva = float(sub['sn_mva'].iloc[0])
                vk_pct = float(sub['vk_percent'].iloc[0])
                vkr_pct = float(sub['vkr_percent'].iloc[0])
                r1 = vkr_pct / 100.0
                z1 = vk_pct / 100.0
                x1 = np.sqrt(max(z1**2 - r1**2, 0))
                # Get HV phases from from_phase column
                hv_phases_raw = sorted(sub.xs(int(hv_bus), level='bus')['from_phase'].unique()) if 'from_phase' in sub.columns else [1,2,3]
                hv_phases = [int(p) - 1 for p in hv_phases_raw]  # 1->0(A), 2->1(B), 3->2(C)
                # Determine PhaseType
                phase_map = {
                    (0,): PhaseType.A, (1,): PhaseType.B, (2,): PhaseType.C,
                    (0,1): PhaseType.AB, (1,2): PhaseType.BC, (0,2): PhaseType.CA,
                    (0,1,2): PhaseType.ABC,
                }
                pt = phase_map.get(tuple(sorted(hv_phases)), PhaseType.ABC)
                t1ph_name = str(sub['name'].iloc[0]) if 'name' in sub.columns and str(sub['name'].iloc[0]) != 'nan' else ''
                net.add_transformer(Transformer(
                    id=f"Trafo1ph_{t_id}", name=t1ph_name,
                    from_bus=hv_bus, to_bus=lv_bus, phase_type=pt,
                    r1=r1, x1=x1, r0=r1, x0=x1,
                    mva_rating=sn_mva,
                    conn_primary='wye_grounded', conn_secondary='wye_grounded',
                    hv_phases=hv_phases
                ))

    # 10. Process asymmetric shunts
    if 'asymmetric_shunt' in pp_net and not pp_net.asymmetric_shunt.empty:
        ashunt = pp_net.asymmetric_shunt
        if isinstance(ashunt.index, pd.MultiIndex):
            idx_name = ashunt.index.names[0]
            for s_id in ashunt.index.get_level_values(idx_name).unique():
                sub = ashunt.loc[s_id] if isinstance(ashunt.loc[s_id], pd.DataFrame) else ashunt.loc[[s_id]]
                b_id = str(int(sub['bus'].iloc[0]))
                p_mw = np.zeros(3)
                q_mvar = np.zeros(3)
                for _, row in sub.iterrows():
                    phase = int(row['from_phase']) - 1 if 'from_phase' in row and not np.isnan(row['from_phase']) else 0
                    if 0 <= phase <= 2:
                        p_mw[phase] += float(row.get('p_mw', 0) or 0)
                        q_mvar[phase] += float(row.get('q_mvar', 0) or 0)
                vn_kv = float(sub['vn_kv'].iloc[0]) if 'vn_kv' in sub.columns else 12.47
                closed = bool(sub['closed'].iloc[0]) if 'closed' in sub.columns else True
                shunt_name = str(sub['name'].iloc[0]) if 'name' in sub.columns and str(sub['name'].iloc[0]) != 'nan' else ''
                net.add_shunt(Shunt(
                    id=f"Shunt_{s_id}", name=shunt_name,
                    bus_id=b_id, phase_type=PhaseType.ABC,
                    p_mw=p_mw, q_mvar=q_mvar,
                    vn_kv=vn_kv, closed=closed
                ))
        else:
            for idx, row in ashunt.iterrows():
                b_id = str(int(row.bus))
                s_id = idx[0] if isinstance(idx, tuple) else idx
                p_mw = np.zeros(3)
                q_mvar = np.zeros(3)
                phase = int(row['from_phase']) - 1 if 'from_phase' in row and not np.isnan(row['from_phase']) else 0
                if 0 <= phase <= 2:
                    p_mw[phase] = float(row.get('p_mw', 0) or 0)
                    q_mvar[phase] = float(row.get('q_mvar', 0) or 0)
                shunt_name2 = str(row['name']) if 'name' in row and str(row['name']) != 'nan' else ''
                net.add_shunt(Shunt(
                    id=f"Shunt_{s_id}", name=shunt_name2,
                    bus_id=b_id, phase_type=PhaseType.ABC,
                    p_mw=p_mw, q_mvar=q_mvar
                ))

    # 11. Process switches — closed bus-bus switches become low-impedance branches
    if 'switch' in pp_net and not pp_net.switch.empty:
        sw_df = pp_net.switch
        if isinstance(sw_df.index, pd.MultiIndex):
            idx_name = sw_df.index.names[0]
            for sw_id in sw_df.index.get_level_values(idx_name).unique():
                sub = sw_df.loc[sw_id] if isinstance(sw_df.loc[sw_id], pd.DataFrame) else sw_df.loc[[sw_id]]
                row = sub.iloc[0]
                sw_bus = int(row['bus'])
                sw_elem = int(row['element'])
                sw_et = str(row.get('et', 'b'))
                sw_closed = bool(row.get('closed', True))
                
                net.add_switch(Switch(
                    id=f"Switch_{sw_id}", bus=sw_bus, element=sw_elem,
                    et=sw_et, sw_type=str(row.get('type', 'LBS')),
                    closed=sw_closed, phase=int(row.get('phase', 0)),
                    r_ohm=float(row.get('r_ohm', 0.0))
                ))
                
                # Closed bus-bus switches: create a low-impedance branch
                if sw_et == 'b' and sw_closed:
                    f_bus = str(sw_bus)
                    t_bus = str(sw_elem)
                    if f_bus in net.buses and t_bus in net.buses:
                        # Determine phases from switch conductors
                        if 'phase' in sub.columns:
                            raw_ph = sorted(sub['phase'].unique())
                            power_phases = sorted([int(p)-1 for p in raw_ph if int(p) >= 1 and int(p) <= 3])
                        else:
                            power_phases = [0, 1, 2]
                        if not power_phases:
                            power_phases = [0, 1, 2]
                        phase_map_sw = {
                            (0,): PhaseType.A, (1,): PhaseType.B, (2,): PhaseType.C,
                            (0,1): PhaseType.AB, (1,2): PhaseType.BC, (0,2): PhaseType.CA,
                            (0,1,2): PhaseType.ABC,
                        }
                        pt = phase_map_sw.get(tuple(power_phases), PhaseType.ABC)
                        # Use small impedance for closed switch (0.001 pu)
                        r_sw = float(row.get('r_ohm', 0.0))
                        f_vk_kv = bus_vk.get(f_bus, 12.47)
                        Zbase = f_vk_kv**2 / base_mva
                        if r_sw > 0:
                            r_pu = r_sw / Zbase
                        else:
                            r_pu = 1e-4
                        net.add_branch(Branch(
                            id=f"SwBr_{sw_id}", from_bus=f_bus, to_bus=t_bus, phase_type=pt,
                            r1=r_pu, x1=r_pu, r0=r_pu, x0=r_pu,
                            ampacity=np.full(3, 9999.0)
                        ))
        else:
            for idx, row in sw_df.iterrows():
                sw_id = idx[0] if isinstance(idx, tuple) else idx
                sw_bus = int(row['bus'])
                sw_elem = int(row['element'])
                sw_et = str(row.get('et', 'b'))
                sw_closed = bool(row.get('closed', True))
                net.add_switch(Switch(
                    id=f"Switch_{sw_id}", bus=sw_bus, element=sw_elem,
                    et=sw_et, sw_type=str(row.get('type', 'LBS')),
                    closed=sw_closed, phase=int(row.get('phase', 0)),
                    r_ohm=float(row.get('r_ohm', 0.0))
                ))
                if sw_et == 'b' and sw_closed:
                    f_bus = str(sw_bus); t_bus = str(sw_elem)
                    if f_bus in net.buses and t_bus in net.buses:
                        f_vk_kv = bus_vk.get(f_bus, 12.47)
                        Zbase = f_vk_kv**2 / base_mva
                        r_sw = float(row.get('r_ohm', 0.0))
                        r_pu = r_sw / Zbase if r_sw > 0 else 1e-4
                        net.add_branch(Branch(
                            id=f"SwBr_{sw_id}", from_bus=f_bus, to_bus=t_bus, phase_type=PhaseType.ABC,
                            r1=r_pu, x1=r_pu, r0=r_pu, x0=r_pu,
                            ampacity=np.full(3, 9999.0)
                        ))

    # 12. Process ext_grid_sequence (if ext_grid is empty)
    if ('ext_grid' not in pp_net or pp_net.ext_grid.empty) and 'ext_grid_sequence' in pp_net and not pp_net.ext_grid_sequence.empty:
        egs = pp_net.ext_grid_sequence
        # Group by first index level to get the ext_grid entry
        if isinstance(egs.index, pd.MultiIndex):
            idx_name = egs.index.names[0]
            for eg_id in egs.index.get_level_values(idx_name).unique():
                sub = egs.loc[eg_id] if isinstance(egs.loc[eg_id], pd.DataFrame) else egs.loc[[eg_id]]
                bus_id = str(int(sub['bus'].iloc[0]))
                # Positive sequence (index 1) has the vm_pu
                pos_seq = sub.iloc[1] if len(sub) > 1 else sub.iloc[0]
                vm_pu = float(pos_seq['vm_pu']) if not np.isnan(pos_seq['vm_pu']) else 1.0
                va_degree = float(pos_seq['va_degree']) if not np.isnan(pos_seq['va_degree']) else 0.0
                if bus_id in net.buses:
                    b = net.buses[bus_id]
                    b.bus_type = BusType.SLACK
                    b.ini_v_mag = np.full(3, vm_pu)
                    b.ini_v_ang = np.array([va_degree, va_degree - 120, va_degree + 120])
                    b.v_mag = b.ini_v_mag.copy()
                    b.v_ang = b.ini_v_ang.copy()
        else:
            for idx, row in egs.iterrows():
                bus_id = str(int(row.bus))
                vm_pu = float(row.vm_pu) if not np.isnan(row.vm_pu) else 1.0
                va_degree = float(row.va_degree) if 'va_degree' in row and not np.isnan(row.va_degree) else 0.0
                if bus_id in net.buses:
                    b = net.buses[bus_id]
                    b.bus_type = BusType.SLACK
                    b.ini_v_mag = np.full(3, vm_pu)
                    b.ini_v_ang = np.array([va_degree, va_degree - 120, va_degree + 120])
                    b.v_mag = b.ini_v_mag.copy()
                    b.v_ang = b.ini_v_ang.copy()

    return net