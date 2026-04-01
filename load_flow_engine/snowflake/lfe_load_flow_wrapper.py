import logging
import pickle
import time

import pandas as pd
import snowflake.snowpark

from load_flow_engine import ThreePhaseLoadFlowSolver
from load_flow_engine.tools.output import extract_res_bus
from powerflow_pipeline.util import PowerflowConfig
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
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_COLS = [
    "CIRCUIT_KEY",
    "NETWORK_BLOB",
]


OUTPUT_FIELDS = [
    StructField("CIRCUIT_KEY", StringType(), nullable=False),
    StructField("BUS_ID", StringType(), nullable=True),
    StructField("BUS_NAME", StringType(), nullable=True),
    StructField("VM_PU", DoubleType(), nullable=True),
    StructField("VA_DEGREE", DoubleType(), nullable=True),
    StructField("V_A_PU", DoubleType(), nullable=True),
    StructField("V_B_PU", DoubleType(), nullable=True),
    StructField("V_C_PU", DoubleType(), nullable=True),
    StructField("VA_A_DEGREE", DoubleType(), nullable=True),
    StructField("VA_B_DEGREE", DoubleType(), nullable=True),
    StructField("VA_C_DEGREE", DoubleType(), nullable=True),
    StructField("CONVERGED", BooleanType(), nullable=False),
    StructField("ITERATIONS", IntegerType(), nullable=True),
    StructField("SOLVE_TIME_SEC", DoubleType(), nullable=True),
    StructField("ERROR_MESSAGE", StringType(), nullable=True),
]


class LFELoadFlow:
    def __init__(
        self,
        session: snowflake.snowpark.Session,
        pf_config_json: str,
    ):
        self.session = session
        self.pf_config_json = pf_config_json
        self.pf_config = PowerflowConfig()
        self.pf_config.load_config_from_json(pf_config_json)

    def register_udtf(self, function_name: str = "LFE_LOAD_FLOW") -> str:
        stage_location = (
            f"{self.pf_config.get_database()}."
            f"{self.pf_config.get_database_schema()}."
            f"{self.pf_config.get_stage_name()}"
        )
        qualified_name = (
            function_name
            if "." in function_name
            else f"{self.pf_config.get_database()}.{self.pf_config.get_database_schema()}.{function_name}"
        )

        @pandas_udtf(
            packages=(
                "snowflake-snowpark-python",
                "numpy==1.26.4",
                "pandas==2.3.2",
                "scipy==1.15.3",
                "typing_extensions==4.15.0",
            ),
            name=qualified_name,
            replace=True,
            stage_location=stage_location,
            output_schema=StructType(OUTPUT_FIELDS),
            input_types=[
                PandasDataFrameType([
                    StringType(),
                    BinaryType(),
                ])
            ],
            input_names=INPUT_COLS,
            statement_params={"STATEMENT_TIMEOUT_IN_SECONDS": 7200},
            is_permanent=True,
        )
        class LFELoadFlowUDTF:
            def end_partition(self, df: PandasDataFrame) -> pd.DataFrame:
                circuit_key = None
                try:
                    if df.empty:
                        return pd.DataFrame(columns=[field.name for field in OUTPUT_FIELDS])

                    circuit_series = df["CIRCUIT_KEY"].dropna()
                    if not circuit_series.empty:
                        circuit_key = str(circuit_series.iloc[0])

                    blob_series = df["NETWORK_BLOB"].dropna()
                    if blob_series.empty:
                        return pd.DataFrame([
                            {
                                "CIRCUIT_KEY": circuit_key or "",
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
                                "ITERATIONS": None,
                                "SOLVE_TIME_SEC": None,
                                "ERROR_MESSAGE": "NETWORK_BLOB is null for this partition",
                            }
                        ])

                    net = pickle.loads(blob_series.iloc[0])
                    solver = ThreePhaseLoadFlowSolver(net)
                    solver.method = 'gs'
                    started = time.perf_counter()
                    converged = bool(solver.solve())
                    solve_time_sec = time.perf_counter() - started
                    iterations = getattr(solver, "_iterations", None)

                    res_bus = extract_res_bus(net).reset_index()
                    res_bus["CIRCUIT_KEY"] = circuit_key or ""
                    res_bus["BUS_ID"] = res_bus["bus"].astype(str)
                    res_bus["BUS_NAME"] = res_bus["name"].fillna("")
                    res_bus["CONVERGED"] = converged
                    res_bus["ITERATIONS"] = iterations
                    res_bus["SOLVE_TIME_SEC"] = solve_time_sec
                    res_bus["ERROR_MESSAGE"] = None

                    return res_bus.rename(
                        columns={
                            "vm_pu": "VM_PU",
                            "va_degree": "VA_DEGREE",
                            "v_a_pu": "V_A_PU",
                            "v_b_pu": "V_B_PU",
                            "v_c_pu": "V_C_PU",
                            "va_a_degree": "VA_A_DEGREE",
                            "va_b_degree": "VA_B_DEGREE",
                            "va_c_degree": "VA_C_DEGREE",
                        }
                    )[
                        [field.name for field in OUTPUT_FIELDS]
                    ]
                except Exception as exc:
                    logger.exception("LFELoadFlowUDTF failed for circuit %s", circuit_key)
                    return pd.DataFrame([
                        {
                            "CIRCUIT_KEY": circuit_key or "",
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
                            "ITERATIONS": None,
                            "SOLVE_TIME_SEC": None,
                            "ERROR_MESSAGE": str(exc),
                        }
                    ])

        logger.info("Registered LFE load flow UDTF: %s", qualified_name)
        return qualified_name