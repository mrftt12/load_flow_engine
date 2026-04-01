-- Output table for LFE_LOAD_FLOW UDTF results.
CREATE TABLE IF NOT EXISTS TESTDB.PUBLIC.LFE_LOAD_FLOW_RESULTS (
    CIRCUIT_KEY     VARCHAR NOT NULL,
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
    CONVERGED       BOOLEAN NOT NULL,
    ITERATIONS      INTEGER,
    SOLVE_TIME_SEC  DOUBLE,
    ERROR_MESSAGE   VARCHAR
);

-- Register the permanent Snowflake Python UDTF that runs load flow on a pickled
-- load_flow_engine.Network blob and returns per-bus voltage results.

CALL TESTDB.PUBLIC.SP_REGISTER_LFE_LOAD_FLOW_UDTF(
    '"defaults": {
                "SNOWFLAKE_ACCOUNT": "ZXGHNCG-ITRON",
                "DATABASE": "TESTDB",
                "DATABASE_SCHEMA": "PUBLIC",
                "CONNECTIVITY_TABLE": "NMM_D_TRACED_CIRCUIT_CONNECTIVITY_C_PP_VW_MC3",
                "BUS_TABLE": "NMM_D_BUS_C_PP_VW_MC3",
                "LOAD_TABLE": "NMM_F_HIST_HR_GROSS_DERGEN_DLY_C_PP_VW_MC3",
                "ENCODED_CIRCUITS_TABLE": "NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C",
                "OUTPUT_TABLE": "NMM_F_TOPOLOGICALNODE_C",
                "DEFINE_CIRCUITS_PROC": "DEFINE_CIRCUITS_PROC",
                "RUN_POWERFLOW_PROC": "RUN_POWERFLOW_PROC",
                "REGISTER_LFE_LOAD_FLOW_UDTF_PROC": "REGISTER_LFE_LOAD_FLOW_UDTF_PROC",
                "STAGE_NAME": "SERVICE_STAGE"
            }',
    'TESTDB.PUBLIC.LFE_LOAD_FLOW'
);

-- Example source query shape expected by the UDTF.
-- NETWORK_BLOB must be a pickled load_flow_engine.Network object.

SELECT
    lfe.CIRCUIT_KEY,
    lfe.BUS_ID,
    lfe.BUS_NAME,
    lfe.VM_PU,
    lfe.VA_DEGREE,
    lfe.V_A_PU,
    lfe.V_B_PU,
    lfe.V_C_PU,
    lfe.VA_A_DEGREE,
    lfe.VA_B_DEGREE,
    lfe.VA_C_DEGREE,
    lfe.CONVERGED,
    lfe.ITERATIONS,
    lfe.SOLVE_TIME_SEC,
    lfe.ERROR_MESSAGE
FROM (
    SELECT
        CIRCUIT_KEY,
        PRELOAD_ENCODED_CA AS NETWORK_BLOB
    FROM TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C
    WHERE CIRCUIT_KEY = 'CKT_114_16955'
) src,
TABLE(TESTDB.PUBLIC.LFE_LOAD_FLOW(src.CIRCUIT_KEY, src.NETWORK_BLOB) OVER (PARTITION BY src.CIRCUIT_KEY)) lfe;


execute NOTEBOOK GRIDMOD_DEV_TD.UC_POC."Ray example"() MAIN_FILE='Ray example.ipynb'
RUNTIME_NAME='SYSTEM$BASIC_RUNTIME'
COMPUTE_POOL=RAY_POC
QUERY_WAREHOUSE=TEST_SMALL