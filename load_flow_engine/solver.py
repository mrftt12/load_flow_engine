"""
Three-phase unbalanced load-flow solver.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, gmres, spilu, splu, LinearOperator

from .enums import PhaseType, BusType
from .constants import _Ai
from .models import BranchResult
from .helpers import _active_phases
from .network import Network


class ThreePhaseLoadFlowSolver:
    """
    Three-phase unbalanced Newton-Raphson load flow solver.

    Algorithmic mapping to ETAP components:
      - State vector: per-phase |V| and ∠V for every non-slack phase node
      - Flat start: 1.0 pu at A=0°, B=−120°, C=+120° (m_IniVMag/Ang defaults)
      - Power mismatch: ΔP, ΔQ per phase per bus  (solution_precision in MVA)
      - Jacobian: assembled analytically from the complex power formulation
      - Voltage update: polar form  Δθ (rad) and Δ|V|/|V| (normalised)
      - Results: written back to Bus.v_mag / Bus.v_ang (m_VMagA/B/C, m_VAngA/B/C)
    """

    def __init__(self, network: Network, method: str = 'nr'):
        """
        Parameters
        ----------
        network : Network
            Built network with Y_abc assembled.
        method : str
            Solver method: 'nr' for Newton-Raphson, 'gs' for Gauss-Seidel.
        """
        self.net  = network
        self.sc   = network.study_case
        self.method      = method.lower()
        self._converged  = False
        self._iterations = 0
        self._V: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> bool:
        """
        Run the load flow solver using the configured method.
        Returns True if the solution converged within max_iterations.
        """
        if self.method == 'gs':
            return self._solve_gauss_seidel()
        return self._solve_nr()

    def _solve_gauss_seidel(self) -> bool:
        """
        Current-injection iterative solver for three-phase unbalanced load flow.

        Uses the implicit Z-bus method: pre-factorizes Y_ff (free-node
        sub-matrix) with sparse LU, then iterates:
            1. I_inj = conj(S_sch) / conj(V)   (load current injections)
            2. rhs   = I_inj[free] - Y_fs @ V_slack
            3. V[free] = Y_ff^{-1} @ rhs       (via pre-factored LU)

        Fully vectorized — no Python loops over bus-phases. Robust for
        ill-conditioned networks with very low impedance switch branches.
        """
        net = self.net
        if net.Y_abc is None:
            net.build()

        buses = net._bus_order
        N     = len(buses)

        V = self._initialise_voltages(buses, N)
        S_sch = self._scheduled_injections(buses, N)

        # Identify slack and free indices
        slack_set = set()
        for i, bid in enumerate(buses):
            if net.buses[bid].bus_type == BusType.SLACK:
                slack_set.update([3*i, 3*i+1, 3*i+2])

        # BFS reachability from slack
        adj: Dict[int, List[int]] = {k: [] for k in range(3 * N)}
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

        Y_diag = np.array(net.Y_abc.diagonal()).flatten()
        free_idx = np.array(sorted(k for k in reachable
                                   if k not in slack_set
                                   and abs(Y_diag[k]) > 1e-10), dtype=int)
        slack_idx = np.array(sorted(slack_set), dtype=int)
        n_free = len(free_idx)

        if n_free == 0:
            self._converged = True
            self._iterations = 0
            self._V = V
            self._write_bus_results(V, buses)
            return True

        # Extract sub-matrices: Y_ff (free×free) and Y_fs (free×slack)
        Y_csc = csc_matrix(net.Y_abc)
        Y_ff = Y_csc[free_idx][:, free_idx]
        Y_fs = Y_csc[free_idx][:, slack_idx]

        # Pre-factorize Y_ff with sparse LU for fast repeated solves
        try:
            lu = splu(Y_ff)
        except Exception:
            warnings.warn("LU factorization of Y_ff failed — falling back to spsolve.")
            lu = None

        V_slack = V[slack_idx]
        rhs_slack = Y_fs @ V_slack  # constant contribution from slack buses

        # Current-injection iteration
        for iteration in range(self.sc.max_iterations):
            # Load current injections at free nodes
            I_inj_free = np.conj(S_sch[free_idx]) / np.conj(V[free_idx])

            # Right-hand side: load current minus slack coupling
            rhs = I_inj_free - rhs_slack

            # Solve for free-node voltages
            if lu is not None:
                V_free_new = lu.solve(rhs)
            else:
                V_free_new = spsolve(Y_ff, rhs)

            # Check convergence via max voltage change
            max_dv = np.max(np.abs(V_free_new - V[free_idx]))
            V[free_idx] = V_free_new

            if max_dv < 1e-6:
                # Verify with power mismatch
                I_calc = np.array(Y_csc @ V).flatten()
                S_calc = V * np.conj(I_calc)
                max_mis = np.max(np.abs(S_sch[free_idx] - S_calc[free_idx]))
                max_mis_mva = max_mis * self.sc.base_mva
                if max_mis_mva < self.sc.solution_precision:
                    self._converged = True
                    self._iterations = iteration + 1
                    break

        else:
            self._iterations = self.sc.max_iterations

        self._V = V
        self._write_bus_results(V, buses)
        self._iterations = self._iterations or self.sc.max_iterations
        return self._converged

    def _solve_nr(self) -> bool:
        """
        Run the Newton-Raphson iteration.
        Returns True if the solution converged within max_iterations.
        """
        net = self.net
        if net.Y_abc is None:
            net.build()

        buses = net._bus_order
        N     = len(buses)

        # ---- Initialise voltage vector (flat start or user-supplied) ----
        V = self._initialise_voltages(buses, N)

        # ---- Identify free (non-slack) node indices ----------------------
        slack_set = set()
        for i, bid in enumerate(buses):
            if net.buses[bid].bus_type == BusType.SLACK:
                slack_set.update([3*i, 3*i+1, 3*i+2])

        # BFS reachability from slack bus
        adj: Dict[int, List[int]] = {k: [] for k in range(3 * N)}
        for br in net.branches.values():
            fi = net._bus_index[br.from_bus]
            ti = net._bus_index[br.to_bus]
            mask = _active_phases(br.phase_type)
            for p in mask:
                a, b_node = 3 * fi + p, 3 * ti + p
                adj[a].append(b_node)
                adj[b_node].append(a)
            # Within each bus, connect active phases to each other
            f_nodes = [3 * fi + p for p in mask]
            t_nodes = [3 * ti + p for p in mask]
            for nodes in (f_nodes, t_nodes):
                for ii in range(len(nodes)):
                    for jj in range(ii + 1, len(nodes)):
                        adj[nodes[ii]].append(nodes[jj])
                        adj[nodes[jj]].append(nodes[ii])
        for xf in net.transformers.values():
            fi = net._bus_index[xf.from_bus]
            ti = net._bus_index[xf.to_bus]
            mask = _active_phases(xf.phase_type)
            for p in mask:
                a, b_node = 3 * fi + p, 3 * ti + p
                adj[a].append(b_node)
                adj[b_node].append(a)
            f_nodes = [3 * fi + p for p in mask]
            t_nodes = [3 * ti + p for p in mask]
            for nodes in (f_nodes, t_nodes):
                for ii in range(len(nodes)):
                    for jj in range(ii + 1, len(nodes)):
                        adj[nodes[ii]].append(nodes[jj])
                        adj[nodes[jj]].append(nodes[ii])

        reachable = set()
        queue = list(slack_set)
        reachable.update(queue)
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if nb not in reachable:
                    reachable.add(nb)
                    queue.append(nb)

        # Diagonal accessor works for both dense and sparse Y_abc
        Y_diag = np.array(net.Y_abc.diagonal()).flatten()
        free_idx = sorted(k for k in reachable
                          if k not in slack_set
                          and abs(Y_diag[k]) > 1e-10)
        free_idx = np.array(free_idx, dtype=int)

        # Build a mapping from global index → local free index
        g2l = np.full(3 * N, -1, dtype=int)
        for loc, glb in enumerate(free_idx):
            g2l[glb] = loc
        n_free = len(free_idx)

        # ---- Scheduled net power injections (pu) -------------------------
        S_sch = self._scheduled_injections(buses, N)

        # ---- Newton-Raphson loop -----------------------------------------
        Y_csc = csc_matrix(net.Y_abc)

        for iteration in range(self.sc.max_iterations):

            I_calc = np.array(Y_csc @ V).flatten()  # I = Y · V
            S_calc = V * np.conj(I_calc)             # S = V ⊙ conj(I)

            dS        = S_sch - S_calc               # complex mismatch
            mismatch  = np.concatenate([
                dS[free_idx].real,
                dS[free_idx].imag])

            max_mis_mva = np.max(np.abs(mismatch)) * self.sc.base_mva
            if max_mis_mva < self.sc.solution_precision:
                self._converged  = True
                self._iterations = iteration + 1
                break

            # Build sparse Jacobian
            J = self._build_jacobian_sparse(V, Y_csc, free_idx, g2l,
                                            n_free, S_calc)
            try:
                # Use ILU preconditioned GMRES for robustness
                try:
                    ilu = spilu(J, drop_tol=1e-6)
                    M = LinearOperator(J.shape, ilu.solve)
                    dx, info = gmres(J, mismatch, M=M, atol=1e-10,
                                     maxiter=200)
                except Exception:
                    dx, info = gmres(J, mismatch, atol=1e-10,
                                     maxiter=500)
                if info != 0:
                    dx = spsolve(J, mismatch)
            except Exception:
                warnings.warn("All sparse solvers failed.")
                self._iterations = iteration + 1
                self._V = V
                return False

            # ---- Polar voltage update ------------------------------------
            dTheta = dx[:n_free]
            dV_rel = dx[n_free:]

            # Clamp updates for numerical stability
            dTheta = np.clip(dTheta, -0.3, 0.3)
            dV_rel = np.clip(dV_rel, -0.3, 0.3)

            for k in range(n_free):
                gidx = free_idx[k]
                mag_k = abs(V[gidx])
                ang_k = np.angle(V[gidx])
                V[gidx] = (mag_k * (1.0 + dV_rel[k])) * np.exp(1j * (ang_k + dTheta[k]))

        else:
            self._iterations = self.sc.max_iterations

        self._V = V
        self._write_bus_results(V, buses)
        self._iterations = self._iterations or self.sc.max_iterations
        return self._converged

    # ------------------------------------------------------------------
    # Branch result computation  (mirrors LFSumBranchLF3PHAccessor)
    # ------------------------------------------------------------------

    def compute_branch_results(self) -> Dict[str, BranchResult]:
        """
        Compute per-phase currents, MVA flows, loading percentages, and
        sequence components for every branch.

        Field mapping to LFSumBranchLF3PH.h:
          i_mag_abc  → m_LoadingAmpMagA/B/C
          i_ang_abc  → m_LoadingAmpAngA/B/C
          mva_from   → m_LoadingInMVAA/B/C
          loading_pct→ m_Loading_A/B/C
          i_mag_012  → m_LoadingAmpMag0/1/2
          cuf2       → m_CUF2 (negative-sequence current unbalance %)
          cuf0       → m_CUF0 (zero-sequence current unbalance %)
        """
        if self._V is None:
            raise RuntimeError("Call solve() before compute_branch_results().")
        if not self._converged:
            warnings.warn("Solver did not converge; branch results are from last iterate.")

        results = {}
        net     = self.net
        V       = self._V
        base    = self.sc.base_mva

        for br_id, br in net.branches.items():
            fi   = net._bus_index[br.from_bus]
            ti   = net._bus_index[br.to_bus]
            mask = _active_phases(br.phase_type)

            Vf = V[3*fi : 3*fi+3]
            Vt = V[3*ti : 3*ti+3]

            prim = net._primitive_admittance(br.r1, br.x1, br.r0, br.x0,
                                              br.phase_type, br_id)
            if prim is None:
                continue
            Y_prim, _ = prim

            # Branch current in the active phase sub-space
            dV_sub   = (Vf - Vt)[mask]
            I_sub    = Y_prim @ dV_sub

            # Expand back to full 3-phase vector
            I_abc = np.zeros(3, dtype=complex)
            for k, p in enumerate(mask):
                I_abc[p] = I_sub[k]

            # Base current for loading % —  I_base = S_base / (√3 · V_base)
            I_base_A = (base * 1e6) / (np.sqrt(3) * net.buses[br.from_bus].base_kv * 1e3)
            i_mag    = np.abs(I_abc)   * I_base_A   # amps
            i_ang    = np.rad2deg(np.angle(I_abc))  # degrees

            # Per-phase apparent power (MVA) flowing from the from-bus
            S_from = Vf * np.conj(I_abc) * base
            S_to   = Vt * np.conj(I_abc) * base

            loading_pct = np.zeros(3)
            for p in range(3):
                if br.ampacity[p] > 1e-6:
                    loading_pct[p] = i_mag[p] / br.ampacity[p] * 100.0

            # ---- Sequence components via Fortescue transform ----
            I_012     = _Ai @ I_abc
            i_mag_012 = np.abs(I_012) * I_base_A
            i_ang_012 = np.rad2deg(np.angle(I_012))

            # Current unbalance factors (matches m_CUF2, m_CUF0)
            I_pos = i_mag_012[1]
            cuf2  = (i_mag_012[2] / I_pos * 100.0) if I_pos > 1e-6 else 0.0
            cuf0  = (i_mag_012[0] / I_pos * 100.0) if I_pos > 1e-6 else 0.0

            results[br_id] = BranchResult(
                id          = br_id,
                from_bus    = br.from_bus,
                to_bus      = br.to_bus,
                i_mag_abc   = i_mag,
                i_ang_abc   = i_ang,
                mva_from    = np.abs(S_from),
                mva_to      = np.abs(S_to),
                loading_pct = loading_pct,
                i_mag_012   = i_mag_012,
                i_ang_012   = i_ang_012,
                cuf2        = cuf2,
                cuf0        = cuf0,
            )

        return results

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_bus_results(self) -> None:
        """Print per-phase bus voltage table (magnitude pu, angle degrees)."""
        buses = self.net._bus_order
        W = 72
        print("\n" + "=" * W)
        print(f"{'Bus':<14} {'Ph':<4} {'|V| (pu)':<12} {'∠V (deg)':<12} {'Type':<8}")
        print("=" * W)
        for bid in buses:
            b = self.net.buses[bid]
            for p, ph in enumerate("ABC"):
                marker = " <-- slack" if b.bus_type == BusType.SLACK and p == 0 else ""
                print(f"{bid:<14} {ph:<4} {b.v_mag[p]:<12.5f} {b.v_ang[p]:<12.3f}"
                      f" {b.bus_type.name:<8}{marker}")
        print("=" * W)
        print(f"  Converged : {self._converged}")
        print(f"  Iterations: {self._iterations}")
        print()

    def print_branch_results(self, results: Dict[str, BranchResult]) -> None:
        """Print per-phase branch current and unbalance factor table."""
        W = 84
        print("=" * W)
        print(f"{'Branch':<12} {'From':<10} {'To':<10} "
              f"{'IA (A)':<9} {'IB (A)':<9} {'IC (A)':<9} "
              f"{'CUF2%':<8} {'CUF0%':<8}")
        print("=" * W)
        for r in results.values():
            print(f"{r.id:<12} {r.from_bus:<10} {r.to_bus:<10} "
                  f"{r.i_mag_abc[0]:<9.2f} {r.i_mag_abc[1]:<9.2f} "
                  f"{r.i_mag_abc[2]:<9.2f} "
                  f"{r.cuf2:<8.2f} {r.cuf0:<8.2f}")
        print("=" * W)
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_bus_results(self, V: np.ndarray, buses: List[str]) -> None:
        """Write solved voltages back to Bus objects."""
        for i, bid in enumerate(buses):
            b = self.net.buses[bid]
            for p in range(3):
                b.v_mag[p] = abs(V[3*i+p])
                b.v_ang[p] = np.rad2deg(np.angle(V[3*i+p]))

    def _initialise_voltages(self, buses: List[str], N: int) -> np.ndarray:
        """
        Build the initial complex voltage vector.
        Flat start: 1.0 pu at A=0°, B=−120°, C=+120°.
        Otherwise uses Bus.ini_v_mag / ini_v_ang (m_IniVMagA/B/C, m_IniAngA/B/C).
        """
        V = np.zeros(3 * N, dtype=complex)
        nom_ang = np.deg2rad([0.0, -120.0, 120.0])
        for i, bid in enumerate(buses):
            b = self.net.buses[bid]
            if self.sc.flat_start:
                for p in range(3):
                    V[3*i+p] = 1.0 * np.exp(1j * nom_ang[p])
            else:
                for p in range(3):
                    ang = np.deg2rad(b.ini_v_ang[p])
                    V[3*i+p] = b.ini_v_mag[p] * np.exp(1j * ang)
        return V

    def _scheduled_injections(self, buses: List[str], N: int) -> np.ndarray:
        """
        Build scheduled complex power injection vector S_sch (length 3N) in pu.
        S_sch[3i+p] = (P_gen − P_load + j·(Q_gen − Q_load)) / base_mva
        """
        S   = np.zeros(3 * N, dtype=complex)
        inv = 1.0 / self.sc.base_mva
        for i, bid in enumerate(buses):
            b = self.net.buses[bid]
            for p in range(3):
                P = (b.gen_mw[p]   - b.load_mw[p])   * inv
                Q = (b.gen_mvar[p] - b.load_mvar[p]) * inv
                S[3*i+p] = complex(P, Q)
        return S

    def _build_jacobian_sparse(self, V: np.ndarray, Y_csc: csc_matrix,
                               free_idx: np.ndarray, g2l: np.ndarray,
                               n_free: int,
                               S_calc: np.ndarray) -> csc_matrix:
        """
        Assemble the NR Jacobian as a sparse matrix for the free nodes.

        For each non-zero Y[i,j] where both i and j are in free_idx,
        computes the Jacobian contributions:
            M1_ij = V[i] * conj(Y[i,j]) * conj(V[j])

        Diagonal (i==j):
            dS_dT_ii = j*(S_calc[i] - M1_ii)
            dS_dV_ii = S_calc[i] + M1_ii
        Off-diagonal:
            dS_dT_ij = -j*M1_ij
            dS_dV_ij = M1_ij
        """
        J = lil_matrix((2 * n_free, 2 * n_free), dtype=float)

        # Iterate over non-zero entries of Y_csc
        Y_coo = Y_csc.tocoo()
        for i, j, yij in zip(Y_coo.row, Y_coo.col, Y_coo.data):
            li = g2l[i]
            lj = g2l[j]
            if li < 0 or lj < 0:
                continue

            m1 = V[i] * np.conj(yij) * np.conj(V[j])

            if i == j:
                ds_dt = 1j * (S_calc[i] - m1)
                ds_dv = S_calc[i] + m1
            else:
                ds_dt = -1j * m1
                ds_dv = m1

            J[li, lj]                 += ds_dt.real   # H block
            J[li, n_free + lj]        += ds_dv.real   # N block
            J[n_free + li, lj]        += ds_dt.imag   # M block
            J[n_free + li, n_free+lj] += ds_dv.imag   # L block

        return csc_matrix(J)
