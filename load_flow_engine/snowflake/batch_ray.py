import ray
import snowflake.snowpark as snp

# Ray initialization (handled by SCS)
ray.init()

# Snowflake connection config (use secrets or env vars in production)
SNOWFLAKE_CONFIG = {
    "account": "ZXGHNCG-ITRON",
    "user": "FGONZALES@ITRON.COM",
    "role": "IDE-AAD-REGRID-TEAM",
    "warehouse": "TEST_SMALL",
    "database": "TESTDB",
    "schema": "PUBLIC"
}

@ray.remote
def run_powerflow_batch(batch_keys):
    session = snp.Session.builder.configs(SNOWFLAKE_CONFIG).create()
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
    session.close()
    return f"Processed {len(batch_keys)} circuits."

def main():
    # Connect to Snowflake to get all circuit keys
    session = snp.Session.builder.configs(SNOWFLAKE_CONFIG).create()
    circuit_keys = session.sql("""
        SELECT DISTINCT CIRCUIT_KEY
        FROM TESTDB.PUBLIC.NMM_F_TOPOLOGICALNODE_ENCODED_CIRCUITS_C
    """).collect()
    circuit_keys = [row['CIRCUIT_KEY'] for row in circuit_keys]
    session.close()

    # Split into N batches
    N = 10  # Adjust based on Ray cluster size
    batches = [circuit_keys[i::N] for i in range(N)]

    # Launch Ray tasks
    futures = [run_powerflow_batch.remote(batch) for batch in batches]
    results = ray.get(futures)
    print(results)

if __name__ == "__main__":
    main()