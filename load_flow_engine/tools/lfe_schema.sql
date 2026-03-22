PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');

-- ---------------------------------------------------------------------------
-- Network-level table  (maps to StudyCase + network identity)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS networks (
  network_id        TEXT PRIMARY KEY,
  name              TEXT,
  description       TEXT,
  base_mva          REAL    DEFAULT 100.0,
  max_iterations    INTEGER DEFAULT 100,
  solution_precision REAL   DEFAULT 1e-4,
  flat_start        INTEGER DEFAULT 1,
  created_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
  updated_at        TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Bus  (maps to models.Bus)
-- Array convention:  _a = phase A (index 0), _b = phase B (1), _c = phase C (2)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bus (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  bus_type    INTEGER DEFAULT 2,   -- BusType enum: SLACK=0, PV=1, PQ=2
  phase_type  INTEGER DEFAULT 0,   -- PhaseType enum: ABC=0, A=5, B=6, C=7, AB=8, BC=9, CA=10
  base_kv     REAL    DEFAULT 12.47,
  name        TEXT    DEFAULT '',
  v_mag_a     REAL DEFAULT 1.0,
  v_mag_b     REAL DEFAULT 1.0,
  v_mag_c     REAL DEFAULT 1.0,
  v_ang_a     REAL DEFAULT 0.0,
  v_ang_b     REAL DEFAULT -120.0,
  v_ang_c     REAL DEFAULT 120.0,
  ini_v_mag_a REAL DEFAULT 1.0,
  ini_v_mag_b REAL DEFAULT 1.0,
  ini_v_mag_c REAL DEFAULT 1.0,
  ini_v_ang_a REAL DEFAULT 0.0,
  ini_v_ang_b REAL DEFAULT -120.0,
  ini_v_ang_c REAL DEFAULT 120.0,
  gen_mw_a    REAL DEFAULT 0.0,
  gen_mw_b    REAL DEFAULT 0.0,
  gen_mw_c    REAL DEFAULT 0.0,
  gen_mvar_a  REAL DEFAULT 0.0,
  gen_mvar_b  REAL DEFAULT 0.0,
  gen_mvar_c  REAL DEFAULT 0.0,
  load_mw_a   REAL DEFAULT 0.0,
  load_mw_b   REAL DEFAULT 0.0,
  load_mw_c   REAL DEFAULT 0.0,
  load_mvar_a REAL DEFAULT 0.0,
  load_mvar_b REAL DEFAULT 0.0,
  load_mvar_c REAL DEFAULT 0.0,
  mvar_max    REAL DEFAULT 9999.0,
  mvar_min    REAL DEFAULT -9999.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Branch  (maps to models.Branch)
-- Impedances in per-unit on system base.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS branch (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  from_bus    TEXT NOT NULL,
  to_bus      TEXT NOT NULL,
  phase_type  INTEGER DEFAULT 0,
  name        TEXT    DEFAULT '',
  r1          REAL DEFAULT 0.0,
  x1          REAL DEFAULT 0.0,
  r0          REAL DEFAULT 0.0,
  x0          REAL DEFAULT 0.0,
  b1          REAL DEFAULT 0.0,
  ampacity_a  REAL DEFAULT 9999.0,
  ampacity_b  REAL DEFAULT 9999.0,
  ampacity_c  REAL DEFAULT 9999.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Transformer  (maps to models.Transformer)
-- Impedances in per-unit on transformer MVA base.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transformer (
  network_id      TEXT NOT NULL,
  id              TEXT NOT NULL,
  from_bus        TEXT NOT NULL,
  to_bus          TEXT NOT NULL,
  phase_type      INTEGER DEFAULT 0,
  name            TEXT    DEFAULT '',
  r1              REAL DEFAULT 0.0,
  x1              REAL DEFAULT 0.01,
  r0              REAL DEFAULT 0.0,
  x0              REAL DEFAULT 0.01,
  mva_rating      REAL DEFAULT 1.0,
  tap_primary     REAL DEFAULT 1.0,
  tap_secondary   REAL DEFAULT 1.0,
  conn_primary    TEXT DEFAULT 'wye_grounded',
  conn_secondary  TEXT DEFAULT 'wye_grounded',
  hv_phases       TEXT DEFAULT '0,1,2',
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Load  (maps to models.Load)
-- Per-phase MW and MVAr.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS load (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  bus_id      TEXT NOT NULL,
  phase_type  INTEGER DEFAULT 0,
  name        TEXT    DEFAULT '',
  mw_a        REAL DEFAULT 0.0,
  mw_b        REAL DEFAULT 0.0,
  mw_c        REAL DEFAULT 0.0,
  mvar_a      REAL DEFAULT 0.0,
  mvar_b      REAL DEFAULT 0.0,
  mvar_c      REAL DEFAULT 0.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Generator  (maps to models.Generator)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS generator (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  bus_id      TEXT NOT NULL,
  bus_type    INTEGER DEFAULT 1,   -- typically PV=1
  name        TEXT    DEFAULT '',
  mw_a        REAL DEFAULT 0.0,
  mw_b        REAL DEFAULT 0.0,
  mw_c        REAL DEFAULT 0.0,
  v_set_pu    REAL DEFAULT 1.0,
  mvar_max    REAL DEFAULT 9999.0,
  mvar_min    REAL DEFAULT -9999.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Shunt  (maps to models.Shunt)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shunt (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  bus_id      TEXT NOT NULL,
  phase_type  INTEGER DEFAULT 0,
  name        TEXT    DEFAULT '',
  p_mw_a      REAL DEFAULT 0.0,
  p_mw_b      REAL DEFAULT 0.0,
  p_mw_c      REAL DEFAULT 0.0,
  q_mvar_a    REAL DEFAULT 0.0,
  q_mvar_b    REAL DEFAULT 0.0,
  q_mvar_c    REAL DEFAULT 0.0,
  vn_kv       REAL DEFAULT 12.47,
  closed      INTEGER DEFAULT 1,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Switch  (maps to models.Switch)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS switch (
  network_id  TEXT NOT NULL,
  id          TEXT NOT NULL,
  bus         INTEGER DEFAULT 0,
  element     INTEGER DEFAULT 0,
  et          TEXT    DEFAULT 'b',
  sw_type     TEXT    DEFAULT 'LBS',
  closed      INTEGER DEFAULT 1,
  phase       INTEGER DEFAULT 0,
  r_ohm       REAL    DEFAULT 0.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Branch results  (maps to models.BranchResult)
-- Populated after a successful solve.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS branch_result (
  network_id    TEXT NOT NULL,
  id            TEXT NOT NULL,
  from_bus      TEXT NOT NULL,
  to_bus        TEXT NOT NULL,
  i_mag_a       REAL DEFAULT 0.0,
  i_mag_b       REAL DEFAULT 0.0,
  i_mag_c       REAL DEFAULT 0.0,
  i_ang_a       REAL DEFAULT 0.0,
  i_ang_b       REAL DEFAULT 0.0,
  i_ang_c       REAL DEFAULT 0.0,
  mva_from_a    REAL DEFAULT 0.0,
  mva_from_b    REAL DEFAULT 0.0,
  mva_from_c    REAL DEFAULT 0.0,
  mva_to_a      REAL DEFAULT 0.0,
  mva_to_b      REAL DEFAULT 0.0,
  mva_to_c      REAL DEFAULT 0.0,
  loading_pct_a REAL DEFAULT 0.0,
  loading_pct_b REAL DEFAULT 0.0,
  loading_pct_c REAL DEFAULT 0.0,
  i_mag_0       REAL DEFAULT 0.0,
  i_mag_1       REAL DEFAULT 0.0,
  i_mag_2       REAL DEFAULT 0.0,
  i_ang_0       REAL DEFAULT 0.0,
  i_ang_1       REAL DEFAULT 0.0,
  i_ang_2       REAL DEFAULT 0.0,
  cuf2          REAL DEFAULT 0.0,
  cuf0          REAL DEFAULT 0.0,
  PRIMARY KEY (network_id, id),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);
