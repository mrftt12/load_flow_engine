"""
SQLite adapter for LFE networks.

Serialize / deserialize :class:`~load_flow_engine.network.Network` objects
to a SQLite database using the schema defined in ``lfe_schema.sql``.

Usage::

    from load_flow_engine.tools.sqlite_adapter import export_network, import_network

    export_network(net, "lfe.db", network_id="my_feeder")
    net = import_network("lfe.db", network_id="my_feeder")
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..enums import PhaseType, BusType
from ..models import (
    StudyCase, Bus, Branch, Transformer, Load, Generator, Shunt, Switch,
    BranchResult,
)
from ..network import Network


_SCHEMA_FILE = Path(__file__).resolve().parent / "lfe_schema.sql"


# ---------------------------------------------------------------------------
# Export  (Network → SQLite)
# ---------------------------------------------------------------------------

def export_network(
    network: Network,
    db_path: str | Path,
    network_id: str = "default",
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    branch_results: Optional[Dict[str, BranchResult]] = None,
) -> None:
    """Write an LFE *network* into a SQLite database.

    If *network_id* already exists in the database it is replaced.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA_FILE.read_text(encoding="utf-8"))

        sc = network.study_case
        conn.execute(
            "INSERT OR REPLACE INTO networks "
            "(network_id, name, description, base_mva, max_iterations, "
            " solution_precision, flat_start) "
            "VALUES (?,?,?,?,?,?,?)",
            (network_id, name, description, sc.base_mva,
             sc.max_iterations, sc.solution_precision, int(sc.flat_start)),
        )

        # Clear previous data for this network_id
        for table in ("bus", "branch", "transformer", "load", "generator",
                       "shunt", "switch", "branch_result"):
            conn.execute(f'DELETE FROM "{table}" WHERE network_id = ?',
                         (network_id,))

        # Buses
        for b in network.buses.values():
            conn.execute(
                "INSERT INTO bus "
                "(network_id, id, bus_type, phase_type, base_kv, name, "
                " v_mag_a, v_mag_b, v_mag_c, v_ang_a, v_ang_b, v_ang_c, "
                " ini_v_mag_a, ini_v_mag_b, ini_v_mag_c, "
                " ini_v_ang_a, ini_v_ang_b, ini_v_ang_c, "
                " gen_mw_a, gen_mw_b, gen_mw_c, "
                " gen_mvar_a, gen_mvar_b, gen_mvar_c, "
                " load_mw_a, load_mw_b, load_mw_c, "
                " load_mvar_a, load_mvar_b, load_mvar_c, "
                " mvar_max, mvar_min) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, b.id, int(b.bus_type), int(b.phase_type),
                 b.base_kv, b.name,
                 *b.v_mag.tolist(), *b.v_ang.tolist(),
                 *b.ini_v_mag.tolist(), *b.ini_v_ang.tolist(),
                 *b.gen_mw.tolist(), *b.gen_mvar.tolist(),
                 *b.load_mw.tolist(), *b.load_mvar.tolist(),
                 b.mvar_max, b.mvar_min),
            )

        # Branches
        for br in network.branches.values():
            conn.execute(
                "INSERT INTO branch "
                "(network_id, id, from_bus, to_bus, phase_type, name, "
                " r1, x1, r0, x0, b1, ampacity_a, ampacity_b, ampacity_c) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, br.id, br.from_bus, br.to_bus,
                 int(br.phase_type), br.name,
                 br.r1, br.x1, br.r0, br.x0, br.b1,
                 *br.ampacity.tolist()),
            )

        # Transformers
        for xf in network.transformers.values():
            conn.execute(
                "INSERT INTO transformer "
                "(network_id, id, from_bus, to_bus, phase_type, name, "
                " r1, x1, r0, x0, mva_rating, tap_primary, tap_secondary, "
                " conn_primary, conn_secondary, hv_phases) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, xf.id, xf.from_bus, xf.to_bus,
                 int(xf.phase_type), xf.name,
                 xf.r1, xf.x1, xf.r0, xf.x0, xf.mva_rating,
                 xf.tap_primary, xf.tap_secondary,
                 xf.conn_primary, xf.conn_secondary,
                 ",".join(str(p) for p in xf.hv_phases)),
            )

        # Loads
        for ld in network.loads.values():
            conn.execute(
                "INSERT INTO load "
                "(network_id, id, bus_id, phase_type, name, "
                " mw_a, mw_b, mw_c, mvar_a, mvar_b, mvar_c) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, ld.id, ld.bus_id, int(ld.phase_type), ld.name,
                 *ld.mw.tolist(), *ld.mvar.tolist()),
            )

        # Generators
        for gen in network.generators.values():
            conn.execute(
                "INSERT INTO generator "
                "(network_id, id, bus_id, bus_type, name, "
                " mw_a, mw_b, mw_c, v_set_pu, mvar_max, mvar_min) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, gen.id, gen.bus_id, int(gen.bus_type), gen.name,
                 *gen.mw.tolist(), gen.v_set_pu,
                 gen.mvar_max, gen.mvar_min),
            )

        # Shunts
        for sh in network.shunts.values():
            conn.execute(
                "INSERT INTO shunt "
                "(network_id, id, bus_id, phase_type, name, "
                " p_mw_a, p_mw_b, p_mw_c, q_mvar_a, q_mvar_b, q_mvar_c, "
                " vn_kv, closed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (network_id, sh.id, sh.bus_id, int(sh.phase_type), sh.name,
                 *sh.p_mw.tolist(), *sh.q_mvar.tolist(),
                 sh.vn_kv, int(sh.closed)),
            )

        # Switches
        for sw in network.switches.values():
            conn.execute(
                "INSERT INTO switch "
                "(network_id, id, bus, element, et, sw_type, closed, "
                " phase, r_ohm) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (network_id, sw.id, sw.bus, sw.element, sw.et,
                 sw.sw_type, int(sw.closed), sw.phase, sw.r_ohm),
            )

        # Branch results (optional)
        if branch_results:
            for br in branch_results.values():
                conn.execute(
                    "INSERT INTO branch_result "
                    "(network_id, id, from_bus, to_bus, "
                    " i_mag_a, i_mag_b, i_mag_c, i_ang_a, i_ang_b, i_ang_c, "
                    " mva_from_a, mva_from_b, mva_from_c, "
                    " mva_to_a, mva_to_b, mva_to_c, "
                    " loading_pct_a, loading_pct_b, loading_pct_c, "
                    " i_mag_0, i_mag_1, i_mag_2, "
                    " i_ang_0, i_ang_1, i_ang_2, "
                    " cuf2, cuf0) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (network_id, br.id, br.from_bus, br.to_bus,
                     *br.i_mag_abc.tolist(), *br.i_ang_abc.tolist(),
                     *br.mva_from.tolist(), *br.mva_to.tolist(),
                     *br.loading_pct.tolist(),
                     *br.i_mag_012.tolist(), *br.i_ang_012.tolist(),
                     br.cuf2, br.cuf0),
                )

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Import  (SQLite → Network)
# ---------------------------------------------------------------------------

def import_network(
    db_path: str | Path,
    network_id: str = "default",
) -> Network:
    """Load an LFE network from a SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Study case
        row = conn.execute(
            "SELECT * FROM networks WHERE network_id = ?", (network_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Network '{network_id}' not found in {db_path}")

        sc = StudyCase(
            max_iterations=row["max_iterations"],
            solution_precision=row["solution_precision"],
            base_mva=row["base_mva"],
            flat_start=bool(row["flat_start"]),
        )
        net = Network(sc)

        # Buses
        for r in conn.execute("SELECT * FROM bus WHERE network_id = ?",
                              (network_id,)):
            net.add_bus(Bus(
                id=r["id"],
                bus_type=BusType(r["bus_type"]),
                phase_type=PhaseType(r["phase_type"]),
                base_kv=r["base_kv"],
                name=r["name"] or '',
                v_mag=np.array([r["v_mag_a"], r["v_mag_b"], r["v_mag_c"]]),
                v_ang=np.array([r["v_ang_a"], r["v_ang_b"], r["v_ang_c"]]),
                ini_v_mag=np.array([r["ini_v_mag_a"], r["ini_v_mag_b"],
                                    r["ini_v_mag_c"]]),
                ini_v_ang=np.array([r["ini_v_ang_a"], r["ini_v_ang_b"],
                                    r["ini_v_ang_c"]]),
                gen_mw=np.array([r["gen_mw_a"], r["gen_mw_b"],
                                 r["gen_mw_c"]]),
                gen_mvar=np.array([r["gen_mvar_a"], r["gen_mvar_b"],
                                   r["gen_mvar_c"]]),
                load_mw=np.array([r["load_mw_a"], r["load_mw_b"],
                                  r["load_mw_c"]]),
                load_mvar=np.array([r["load_mvar_a"], r["load_mvar_b"],
                                    r["load_mvar_c"]]),
                mvar_max=r["mvar_max"],
                mvar_min=r["mvar_min"],
            ))

        # Branches
        for r in conn.execute("SELECT * FROM branch WHERE network_id = ?",
                              (network_id,)):
            net.add_branch(Branch(
                id=r["id"],
                from_bus=r["from_bus"],
                to_bus=r["to_bus"],
                phase_type=PhaseType(r["phase_type"]),
                name=r["name"] or '',
                r1=r["r1"], x1=r["x1"], r0=r["r0"], x0=r["x0"],
                b1=r["b1"],
                ampacity=np.array([r["ampacity_a"], r["ampacity_b"],
                                   r["ampacity_c"]]),
            ))

        # Transformers
        for r in conn.execute(
                "SELECT * FROM transformer WHERE network_id = ?",
                (network_id,)):
            hv = [int(x) for x in (r["hv_phases"] or "0,1,2").split(",")]
            net.add_transformer(Transformer(
                id=r["id"],
                from_bus=r["from_bus"],
                to_bus=r["to_bus"],
                phase_type=PhaseType(r["phase_type"]),
                name=r["name"] or '',
                r1=r["r1"], x1=r["x1"], r0=r["r0"], x0=r["x0"],
                mva_rating=r["mva_rating"],
                tap_primary=r["tap_primary"],
                tap_secondary=r["tap_secondary"],
                conn_primary=r["conn_primary"],
                conn_secondary=r["conn_secondary"],
                hv_phases=hv,
            ))

        # Loads
        for r in conn.execute("SELECT * FROM load WHERE network_id = ?",
                              (network_id,)):
            net.add_load(Load(
                id=r["id"],
                bus_id=r["bus_id"],
                phase_type=PhaseType(r["phase_type"]),
                name=r["name"] or '',
                mw=np.array([r["mw_a"], r["mw_b"], r["mw_c"]]),
                mvar=np.array([r["mvar_a"], r["mvar_b"], r["mvar_c"]]),
            ))

        # Generators
        for r in conn.execute(
                "SELECT * FROM generator WHERE network_id = ?",
                (network_id,)):
            net.add_generator(Generator(
                id=r["id"],
                bus_id=r["bus_id"],
                bus_type=BusType(r["bus_type"]),
                name=r["name"] or '',
                mw=np.array([r["mw_a"], r["mw_b"], r["mw_c"]]),
                v_set_pu=r["v_set_pu"],
                mvar_max=r["mvar_max"],
                mvar_min=r["mvar_min"],
            ))

        # Shunts
        for r in conn.execute("SELECT * FROM shunt WHERE network_id = ?",
                              (network_id,)):
            net.add_shunt(Shunt(
                id=r["id"],
                bus_id=r["bus_id"],
                phase_type=PhaseType(r["phase_type"]),
                name=r["name"] or '',
                p_mw=np.array([r["p_mw_a"], r["p_mw_b"], r["p_mw_c"]]),
                q_mvar=np.array([r["q_mvar_a"], r["q_mvar_b"],
                                 r["q_mvar_c"]]),
                vn_kv=r["vn_kv"],
                closed=bool(r["closed"]),
            ))

        # Switches
        for r in conn.execute("SELECT * FROM switch WHERE network_id = ?",
                              (network_id,)):
            net.add_switch(Switch(
                id=r["id"],
                bus=r["bus"],
                element=r["element"],
                et=r["et"],
                sw_type=r["sw_type"],
                closed=bool(r["closed"]),
                phase=r["phase"],
                r_ohm=r["r_ohm"],
            ))

        return net
    finally:
        conn.close()


def list_networks(db_path: str | Path) -> List[str]:
    """Return a list of network_id values stored in the database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT network_id FROM networks").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def delete_network(db_path: str | Path, network_id: str) -> None:
    """Remove a network and all its element rows from the database."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM networks WHERE network_id = ?",
                      (network_id,))
        conn.commit()
    finally:
        conn.close()
