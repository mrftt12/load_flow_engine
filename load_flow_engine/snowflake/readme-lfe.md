## Snowflake Python UDTF for Load Flow Engine

### What was added

1  New LFE UDTF wrapper in lfe_load_flow_wrapper.py</br>
2.  New registration procedure hook in procedure.py</br>
3.  New config getter/default in util.py</br>
4.  Installer registration in install.py</br>
5.  Example SQL usage in lfe_load_flow_udtf.sql</br>

### What the UDTF does

1.  Accepts a partitioned input with:
    - CIRCUIT_KEY
    - NETWORK_BLOB
2.  Expects NETWORK_BLOB to be a pickled load_flow_engine.Network object
3.  Deserializes the network
4.  Runs ThreePhaseLoadFlowSolver
5.  Extracts per-bus voltage results using the LFE result helper
6.  Returns one row per bus with:
    - circuit key
    - bus id
    - bus name
    - positive-sequence vm/va
    - phase A/B/C magnitudes and angles
    - converged flag
    - iteration count
    - solve time
    - error message

### Important constraint

1.  This UDTF is built around a pickled LFE network blob, not the existing multiconductor pipeline blob used by the MC Snowflake flow.

2.  If your Snowflake encoded-circuit table currently stores pipeline objects or MC objects in PRELOAD_ENCODED_CA, you will need an LFE-specific encoded blob source table, or a conversion step before invoking this function.

### Registration path

1.  I added a stored procedure entry point named from config via REGISTER_LFE_LOAD_FLOW_UDTF_PROC
2.  Default procedure name is now available through util.py
3.  The deploy installer in install.py now includes this registration procedure

### How to use it

1.  Install procedures with the existing deploy flow
2.  Call the registration procedure to create the permanent UDTF
3.  Invoke the UDTF over a table containing CIRCUIT_KEY and NETWORK_BLOB

### Example reference

1.  See lfe_load_flow_udtf.sql for the intended registration and TABLE(...) call pattern

### Validation

1.  I checked the changed Python files for editor-detected errors
2.  No static errors were reported
3.  I did not execute this against a live Snowflake session from this environment