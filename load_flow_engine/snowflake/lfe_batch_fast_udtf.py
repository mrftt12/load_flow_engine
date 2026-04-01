"""
Snowflake UDTF that runs ``run_pf_batch_fast`` on a pickled LFE Network
with time-series load data, distributing work across partitions via a
Snowflake compute pool.

Usage
-----
Call ``LFEBatchFastPowerflow.register_udtf(session, …)`` from a stored
procedure to register the UDTF, then invoke it from SQL (see the
companion ``lfe_batch_fast_udtf.sql``).
"""

import logging
import pickle
import time

import numpy as np
import pandas as pd
import snowflake.snowpark

from load_flow_engine.time_series import run_pf_batch_fast
from snowflake.snowpark.functions import PandasDataFrameType, pandas_udtf
from snowflake.snowpark.types import (
    BinaryType,
    BooleanType,
    DoubleType,
    IntegerType,
    PandasDataFrame,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Schema constants                                                       #
# --------------------------------------------------------------------- #

INPUT_COLS = [
    "CIRCUIT_KEY",          # VARCHAR  – circuit identifier
    "NETWORK_BLOB",         # BINARY   – pickled load_flow_engine.Network
    "REPORTED_DTTM",        # TIMESTAMP – hour timestamp
    "MEASURE_VALUE",        # DOUBLE   – total active power (MW) for this hour
    "PARTITION_ID",         # INTEGER  – which partition this row belongs to
]

OUTPUT_FIELDS = [
    StructField("CIRCUIT_KEY",   StringType(),    nullable=False),
    StructField("PARTITION_ID",  IntegerType(),   nullable=False),
    StructField("REPORTED_DTTM", StringType(),    nullable=True),
    StructField("BUS_ID",        StringType(),    nullable=True),
    StructField("BUS_NAME",      StringType(),    nullable=True),
    StructField("VM_PU",         DoubleType(),    nullable=True),
    StructField("VA_DEGREE",     DoubleType(),    nullable=True),
    StructField("V_A_PU",        DoubleType(),    nullable=True),
    StructField("V_B_PU",        DoubleType(),    nullable=True),
    StructField("V_C_PU",        DoubleType(),    nullable=True),
    StructField("VA_A_DEGREE",   DoubleType(),    nullable=True),
    StructField("VA_B_DEGREE",   DoubleType(),    nullable=True),
    StructField("VA_C_DEGREE",   DoubleType(),    nullable=True),
    StructField("CONVERGED",     BooleanType(),   nullable=False),
    StructField("HR",            IntegerType(),   nullable=True),
    StructField("SOLVE_TIME_SEC", DoubleType(),   nullable=True),
    StructField("ERROR_MESSAGE", StringType(),    nullable=True),
]


# --------------------------------------------------------------------- #
# Registration wrapper                                                   #
# --------------------------------------------------------------------- #

class LFEBatchFastPowerflow:
    """Registers and manages the ``LFE_BATCH_FAST_PF`` Snowflake UDTF."""

    def __init__(
        self,
        session: snowflake.snowpark.Session,
        database: str,
        schema: str,
        stage_name: str,
    ):
        self.session = session
        self.database = database
        self.schema = schema
        self.stage_name = stage_name

    def register_udtf(
        self,
        function_name: str = "LFE_BATCH_FAST_PF",
    ) -> str:
        stage_location = f"{self.database}.{self.schema}.{self.stage_name}"
        qualified_name = (
            function_name
            if "." in function_name
            else f"{self.database}.{self.schema}.{function_name}"
        )

        @pandas_udtf(
            packages=(
                "snowflake-snowpark-python",
                "numpy==1.26.4",
                "pandas==2.3.2",
                "scipy==1.15.3",
                "typing_extensions==4.15.0",
                "geojson",
                "numba",
                "deepdiff",
                "networkx",
                "lxml",
                "tqdm",
            ),
            name=qualified_name,
            replace=True,
            stage_location=stage_location,
            output_schema=StructType(OUTPUT_FIELDS),
            input_types=[
                PandasDataFrameType([
                    StringType(),       # CIRCUIT_KEY
                    BinaryType(),       # NETWORK_BLOB
                    TimestampType(),    # REPORTED_DTTM
                    DoubleType(),       # MEASURE_VALUE
                    IntegerType(),      # PARTITION_ID
                ])
            ],
            input_names=INPUT_COLS,
            statement_params={
                "STATEMENT_TIMEOUT_IN_SECONDS": 7200,
                "PYTHON_UDTF_END_PARTITION_TIMEOUT_SECONDS": 3600,
            },
            is_permanent=True,
        )
        class LFEBatchFastPFUDTF:
            """
            Each partition receives all time-series rows for ONE
            (CIRCUIT_KEY, PARTITION_ID) combination.  The UDTF:

            1. Deserialises the Network blob (same blob repeated per row).
            2. Sorts the load values by timestamp.
            3. Calls ``run_pf_batch_fast`` on the sorted MW series.
            4. Maps the ``hr`` index back to the original timestamps.
            5. Returns per-bus, per-hour voltage results.
            """

            def end_partition(self, df: PandasDataFrame) -> pd.DataFrame:
                circuit_key = ""
                partition_id = 0
                try:
                    if df.empty:
                        return pd.DataFrame(
                            columns=[f.name for f in OUTPUT_FIELDS]
                        )

                    circuit_key = str(
                        df["CIRCUIT_KEY"].dropna().iloc[0]
                    )
                    partition_id = int(
                        df["PARTITION_ID"].dropna().iloc[0]
                    )

                    # --- Deserialise network ---
                    blob = df["NETWORK_BLOB"].dropna().iloc[0]
                    net = pickle.loads(blob)

                    # --- Build ordered MW series ---
                    ts = (
                        df[["REPORTED_DTTM", "MEASURE_VALUE"]]
                        .dropna(subset=["MEASURE_VALUE"])
                        .sort_values("REPORTED_DTTM")
                        .reset_index(drop=True)
                    )

                    if ts.empty:
                        return pd.DataFrame([{
                            "CIRCUIT_KEY": circuit_key,
                            "PARTITION_ID": partition_id,
                            "REPORTED_DTTM": None,
                            "BUS_ID": None,
                            "BUS_NAME": None,
                            "VM_PU": None,
                            "VA_DEGREE": None,
                            "V_A_PU": None,
                            "V_B_PU": None,
                            "V_C_PU": None,
                            "VA_A_DEGREE": None,
                            "VA_B_DEGREE": None,
                            "VA_C_DEGREE": None,
                            "CONVERGED": False,
                            "HR": None,
                            "SOLVE_TIME_SEC": None,
                            "ERROR_MESSAGE": "No valid MEASURE_VALUE rows",
                        }])

                    values = ts["MEASURE_VALUE"].values.astype(float)
                    timestamps = ts["REPORTED_DTTM"].values

                    # --- Run fast batch power flow ---
                    t0 = time.perf_counter()
                    result_df = run_pf_batch_fast(
                        net,
                        values,
                        warm_start=True,
                        power_factor=0.95,
                    )
                    solve_time = time.perf_counter() - t0

                    # --- Map hr back to timestamp ---
                    hr_to_ts = {
                        hr + 1: timestamps[hr]
                        for hr in range(len(timestamps))
                    }
                    result_df = result_df.reset_index()
                    result_df["REPORTED_DTTM"] = (
                        result_df["hr"].map(hr_to_ts).astype(str)
                    )

                    # --- Build output ---
                    result_df["CIRCUIT_KEY"] = circuit_key
                    result_df["PARTITION_ID"] = partition_id
                    result_df["BUS_ID"] = result_df["bus"].astype(str)
                    result_df["BUS_NAME"] = result_df["name"].fillna("")
                    result_df["CONVERGED"] = result_df["converged"]
                    result_df["HR"] = result_df["hr"]
                    result_df["SOLVE_TIME_SEC"] = solve_time
                    result_df["ERROR_MESSAGE"] = None

                    return result_df.rename(columns={
                        "vm_pu":        "VM_PU",
                        "va_degree":    "VA_DEGREE",
                        "v_a_pu":       "V_A_PU",
                        "v_b_pu":       "V_B_PU",
                        "v_c_pu":       "V_C_PU",
                        "va_a_degree":  "VA_A_DEGREE",
                        "va_b_degree":  "VA_B_DEGREE",
                        "va_c_degree":  "VA_C_DEGREE",
                    })[[f.name for f in OUTPUT_FIELDS]]

                except Exception as exc:
                    logger.exception(
                        "LFEBatchFastPFUDTF failed for circuit=%s partition=%s",
                        circuit_key, partition_id,
                    )
                    return pd.DataFrame([{
                        "CIRCUIT_KEY": circuit_key or "",
                        "PARTITION_ID": partition_id,
                        "REPORTED_DTTM": None,
                        "BUS_ID": None,
                        "BUS_NAME": None,
                        "VM_PU": None,
                        "VA_DEGREE": None,
                        "V_A_PU": None,
                        "V_B_PU": None,
                        "V_C_PU": None,
                        "VA_A_DEGREE": None,
                        "VA_B_DEGREE": None,
                        "VA_C_DEGREE": None,
                        "CONVERGED": False,
                        "HR": None,
                        "SOLVE_TIME_SEC": None,
                        "ERROR_MESSAGE": str(exc),
                    }])

        logger.info("Registered LFE batch fast PF UDTF: %s", qualified_name)
        return qualified_name
