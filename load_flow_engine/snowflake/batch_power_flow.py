import snowflake.snowpark as snp
import os
# Connect to Snowflake (use your connection params)
session = snp.Session.builder.configs({
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}).create()

# Get all circuit keys
circuit_keys = session.sql("""
    SELECT DISTINCT CIRCUIT_KEY
    FROM TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C
""").collect()
circuit_keys = [row['CIRCUIT_KEY'] for row in circuit_keys]

# Split into N batches (e.g., 10)
N = 10
batches = [circuit_keys[i::N] for i in range(N)]

def run_batch(batch_keys):
    keys_str = "', '".join(batch_keys)
    query = f"""
        INSERT INTO TESTDB.PUBLIC.LFE_LOAD_FLOW_RESULTS
        SELECT
            lfe.CIRCUIT_KEY, lfe.BUS_ID, lfe.BUS_NAME, lfe.VM_PU,
            lfe.VA_DEGREE, lfe.V_A_PU, lfe.V_B_PU, lfe.V_C_PU, lfe.VA_A_DEGREE, lfe.VA_B_DEGREE, lfe.VA_C_DEGREE,
            lfe.CONVERGED, lfe.ITERATIONS, lfe.SOLVE_TIME_SEC, lfe.ERROR_MESSAGE
        FROM (
            SELECT CIRCUIT_KEY, PRELOAD_ENCODED_CA AS NETWORK_BLOB
            FROM TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C
            WHERE CIRCUIT_KEY IN ('{keys_str}')
        ) src,
        TABLE(TESTDB.PUBLIC.LFE_LOAD_FLOW(src.CIRCUIT_KEY, src.NETWORK_BLOB) OVER (PARTITION BY src.CIRCUIT_KEY)) lfe
    """
    session.sql(query).collect()

# Use concurrent.futures to run in parallel (locally or in a container service)
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
    executor.map(run_batch, batches)