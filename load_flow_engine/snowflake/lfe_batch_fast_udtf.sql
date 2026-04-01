-- ============================================================
-- LFE_BATCH_FAST_PF  –  distributed time-series power flow
-- ============================================================
--
-- Time-series source: TESTDB.PUBLIC.NMM_TS
--   Wide table – one column per circuit (917 circuits).
--   Rows keyed by REPORTED_DTTM.
--   Each circuit column holds the load value in MW.
--
-- Encoded networks: TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C
--   CIRCUIT_KEY + PRELOAD_ENCODED_CA (pickled Network blob).
--
-- N_PARTITIONS splits each circuit's hours into chunks so the
-- UDTF partitions run on separate warehouse/compute-pool nodes.
-- ============================================================


-- 1. Output table ---------------------------------------------------
CREATE TABLE IF NOT EXISTS TESTDB.PUBLIC.LFE_BATCH_FAST_PF_RESULTS (
    CIRCUIT_KEY     VARCHAR  NOT NULL,
    PARTITION_ID    INTEGER  NOT NULL,
    REPORTED_DTTM   VARCHAR,
    BUS_ID          VARCHAR,
    BUS_NAME        VARCHAR,
    VM_PU           DOUBLE,
    VA_DEGREE       DOUBLE,
    V_A_PU          DOUBLE,
    V_B_PU          DOUBLE,
    V_C_PU          DOUBLE,
    VA_A_DEGREE     DOUBLE,
    VA_B_DEGREE     DOUBLE,
    VA_C_DEGREE     DOUBLE,
    CONVERGED       BOOLEAN  NOT NULL,
    HR              INTEGER,
    SOLVE_TIME_SEC  DOUBLE,
    ERROR_MESSAGE   VARCHAR
);


-- 2. Stored procedure: register the Python UDTF -----------------------
--    Run this once (or after code changes) to push the UDTF into
--    the stage and make it callable.

CREATE OR REPLACE PROCEDURE TESTDB.PUBLIC.SP_REGISTER_LFE_BATCH_FAST_PF()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = (
    'snowflake-snowpark-python',
    'numpy==1.26.4',
    'pandas==2.3.2',
    'scipy==1.15.3',
    'typing_extensions==4.15.0',
    'geojson','numba','deepdiff',
    'networkx','lxml','tqdm'
)
IMPORTS = ('@TESTDB.PUBLIC.SERVICE_STAGE/lfe_batch_fast_udtf.py',
           '@TESTDB.PUBLIC.SERVICE_STAGE/load_flow_engine.zip',
           '@TESTDB.PUBLIC.SERVICE_STAGE/pandapower.zip')
HANDLER = 'register_handler'
EXECUTE AS CALLER
AS
$$
def register_handler(session):
    from lfe_batch_fast_udtf import LFEBatchFastPowerflow
    wrapper = LFEBatchFastPowerflow(
        session,
        database="TESTDB",
        schema="PUBLIC",
        stage_name="SERVICE_STAGE",
    )
    return wrapper.register_udtf("TESTDB.PUBLIC.LFE_BATCH_FAST_PF")
$$;

-- Register the UDTF
CALL TESTDB.PUBLIC.SP_REGISTER_LFE_BATCH_FAST_PF();


-- 3. Example: single circuit from the wide NMM_TS table ---------------
--    NMM_TS has one column per circuit, so reference the column directly.

SET n_partitions = 4;

INSERT INTO TESTDB.PUBLIC.LFE_BATCH_FAST_PF_RESULTS
WITH ts_unpivoted AS (
    SELECT
        REPORTED_DTTM::TIMESTAMP_NTZ            AS REPORTED_DTTM,
        'CKT_4799_01085'                        AS CIRCUIT_KEY,
        "CKT_4799_01085"::DOUBLE                AS MEASURE_VALUE
    FROM TESTDB.PUBLIC.NMM_TS
    WHERE "CKT_4799_01085" IS NOT NULL
),
partitioned_ts AS (
    SELECT
        CIRCUIT_KEY,
        REPORTED_DTTM,
        MEASURE_VALUE,
        NTILE($n_partitions) OVER (
            PARTITION BY CIRCUIT_KEY
            ORDER BY REPORTED_DTTM
        )                                       AS PARTITION_ID
    FROM ts_unpivoted
),
source AS (
    SELECT
        pts.CIRCUIT_KEY,
        enc.PRELOAD_ENCODED_CA                  AS NETWORK_BLOB,
        pts.REPORTED_DTTM,
        pts.MEASURE_VALUE,
        pts.PARTITION_ID
    FROM partitioned_ts pts
    JOIN TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C enc
        ON enc.CIRCUIT_KEY = pts.CIRCUIT_KEY
)
SELECT
    pf.CIRCUIT_KEY,
    pf.PARTITION_ID,
    pf.REPORTED_DTTM,
    pf.BUS_ID,
    pf.BUS_NAME,
    pf.VM_PU,
    pf.VA_DEGREE,
    pf.V_A_PU,
    pf.V_B_PU,
    pf.V_C_PU,
    pf.VA_A_DEGREE,
    pf.VA_B_DEGREE,
    pf.VA_C_DEGREE,
    pf.CONVERGED,
    pf.HR,
    pf.SOLVE_TIME_SEC,
    pf.ERROR_MESSAGE
FROM source,
     TABLE(
         TESTDB.PUBLIC.LFE_BATCH_FAST_PF(
             source.CIRCUIT_KEY,
             source.NETWORK_BLOB,
             source.REPORTED_DTTM,
             source.MEASURE_VALUE,
             source.PARTITION_ID
         ) OVER (PARTITION BY source.CIRCUIT_KEY, source.PARTITION_ID)
     ) pf;


-- 4. Stored procedure: run batch fast PF for any / all circuits -------
--    Dynamically unpivots the wide NMM_TS table so every circuit
--    column becomes (CIRCUIT_KEY, REPORTED_DTTM, MEASURE_VALUE) rows.
--    Pass circuit_keys = '*' to run all 917 circuits.

CREATE OR REPLACE PROCEDURE TESTDB.PUBLIC.SP_RUN_BATCH_FAST_PF(
    encoded_table  VARCHAR,
    ts_table       VARCHAR,
    n_partitions   INTEGER,
    circuit_keys   VARCHAR   -- comma-separated list, or '*' for all
)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run_handler'
EXECUTE AS CALLER
AS
$$
def run_handler(session, encoded_table: str, ts_table: str,
                n_partitions: int, circuit_keys: str):
    # ---- discover circuit columns from the wide table ----
    cols_df = session.sql(
        f"""SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'PUBLIC'
              AND TABLE_NAME   = '{ts_table.split('.')[-1]}'
              AND COLUMN_NAME != 'REPORTED_DTTM'
            ORDER BY ORDINAL_POSITION"""
    ).collect()
    all_circuits = [r['COLUMN_NAME'] for r in cols_df]

    if circuit_keys.strip() == '*':
        target_circuits = all_circuits
    else:
        requested = {c.strip().upper() for c in circuit_keys.split(',')}
        target_circuits = [c for c in all_circuits if c in requested]

    if not target_circuits:
        return "ERROR: no matching circuit columns found in " + ts_table

    # ---- build UNPIVOT column list ----
    col_list = ", ".join(f'"{c}"' for c in target_circuits)

    query = f"""
    INSERT INTO TESTDB.PUBLIC.LFE_BATCH_FAST_PF_RESULTS
    WITH ts_unpivoted AS (
        SELECT
            REPORTED_DTTM::TIMESTAMP_NTZ  AS REPORTED_DTTM,
            CIRCUIT_KEY,
            MEASURE_VALUE::DOUBLE         AS MEASURE_VALUE
        FROM {ts_table}
        UNPIVOT(MEASURE_VALUE FOR CIRCUIT_KEY IN ({col_list}))
        WHERE MEASURE_VALUE IS NOT NULL
    ),
    partitioned_ts AS (
        SELECT
            CIRCUIT_KEY,
            REPORTED_DTTM,
            MEASURE_VALUE,
            NTILE({int(n_partitions)}) OVER (
                PARTITION BY CIRCUIT_KEY
                ORDER BY REPORTED_DTTM
            ) AS PARTITION_ID
        FROM ts_unpivoted
    ),
    source AS (
        SELECT
            pts.CIRCUIT_KEY,
            enc.PRELOAD_ENCODED_CA  AS NETWORK_BLOB,
            pts.REPORTED_DTTM,
            pts.MEASURE_VALUE,
            pts.PARTITION_ID
        FROM partitioned_ts pts
        JOIN {encoded_table} enc
            ON enc.CIRCUIT_KEY = pts.CIRCUIT_KEY
    )
    SELECT
        pf.CIRCUIT_KEY,
        pf.PARTITION_ID,
        pf.REPORTED_DTTM,
        pf.BUS_ID,
        pf.BUS_NAME,
        pf.VM_PU,
        pf.VA_DEGREE,
        pf.V_A_PU,
        pf.V_B_PU,
        pf.V_C_PU,
        pf.VA_A_DEGREE,
        pf.VA_B_DEGREE,
        pf.VA_C_DEGREE,
        pf.CONVERGED,
        pf.HR,
        pf.SOLVE_TIME_SEC,
        pf.ERROR_MESSAGE
    FROM source,
         TABLE(
             TESTDB.PUBLIC.LFE_BATCH_FAST_PF(
                 source.CIRCUIT_KEY,
                 source.NETWORK_BLOB,
                 source.REPORTED_DTTM,
                 source.MEASURE_VALUE,
                 source.PARTITION_ID
             ) OVER (PARTITION BY source.CIRCUIT_KEY, source.PARTITION_ID)
         ) pf
    """
    session.sql(query).collect()
    return f"SUCCESS: {len(target_circuits)} circuits x {n_partitions} partitions"
$$;


-- 5. Execute examples -------------------------------------------------

-- Single circuit, 4 partitions:
CALL TESTDB.PUBLIC.SP_RUN_BATCH_FAST_PF(
    'TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C',
    'TESTDB.PUBLIC.NMM_TS',
    4,
    'CKT_4799_01085'
);

-- Multiple circuits:
CALL TESTDB.PUBLIC.SP_RUN_BATCH_FAST_PF(
    'TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C',
    'TESTDB.PUBLIC.NMM_TS',
    4,
    'CKT_4799_01085, CKT_114_16955, CKT_2644_04220'
);

-- All 917 circuits:
CALL TESTDB.PUBLIC.SP_RUN_BATCH_FAST_PF(
    'TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C',
    'TESTDB.PUBLIC.NMM_TS',
    4,
    '*'
);
