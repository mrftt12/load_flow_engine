from dataclasses import dataclass
from functools import wraps
import json
import os
from pathlib import Path
import tempfile
import time
import snowflake.snowpark



def track_time(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        self.metrics.append(f"{self.runid},{func.__name__},{duration}")
        return result

    return wrapper


def validate(validation_func):
    """
    Decorator that runs a validation function on the method before execution.
    Raises an exception if validation fails.

    Args:
        validation_func: A function that takes the same arguments as the decorated method
                        and returns None if validation passes or raises an exception if not.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            validation_func(*args, **kwargs)
            return method(*args, **kwargs)

        return wrapper

    return decorator


class PowerflowConfig:
    def __init__(self):
        self.env = None
        self.stage_name = "SERVICE_STAGE"
        self.config = {
            "defaults": {
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
            }
        }

    def get_config_value(self, key):
        return (
            self.config[self.env][key]
            if self.env in self.config and key in self.config[self.env]
            else self.config["defaults"][key]
        )
        
    def get_active_config(self):
        active_config  = self.config[self.env] if self.env in self.config else self.config["defaults"]
        return active_config

    def get_stage_name(self):
        return self.stage_name

    def get_database(self):
        return self.get_config_value("DATABASE")

    def get_database_schema(self):
        return self.get_config_value("DATABASE_SCHEMA")
    
    def get_connectivity_table_database(self):
        return (
            self.get_config_value("CONNECTIVITY_TABLE_DATABASE")
            if "CONNECTIVITY_TABLE_DATABASE" in self.get_active_config()
            else self.get_config_value("DATABASE")
        )  
            
    def get_connectivity_table_schema(self):
        return (
            self.get_config_value("CONNECTIVITY_TABLE_SCHEMA")
            if "CONNECTIVITY_TABLE_SCHEMA" in self.get_active_config()
            else self.get_config_value("DATABASE_SCHEMA")
        )    

    def get_connectivity_table(self):
        return self.get_config_value("CONNECTIVITY_TABLE")

    def get_bus_table_database(self):
        return (
            self.get_config_value("BUS_TABLE_DATABASE")
            if "BUS_TABLE_DATABASE" in self.get_active_config()
            else self.get_config_value("DATABASE")
        )
        
    def get_bus_table_schema(self):
        return (
            self.get_config_value("BUS_TABLE_SCHEMA")
            if "BUS_TABLE_SCHEMA" in self.get_active_config()
            else self.get_config_value("DATABASE_SCHEMA")
        )

    def get_bus_table(self):
        return self.get_config_value("BUS_TABLE")

    def get_load_table_database(self):
        return (
            self.get_config_value("LOAD_TABLE_DATABASE")
            if "LOAD_TABLE_DATABASE" in self.get_active_config()
            else self.get_config_value("DATABASE")
        )
        
    def get_load_table_schema(self):
        return (
            self.get_config_value("LOAD_TABLE_SCHEMA")
            if "LOAD_TABLE_SCHEMA" in self.get_active_config()
            else self.get_config_value("DATABASE_SCHEMA")
        )
        
    def get_load_table(self):
        return self.get_config_value("LOAD_TABLE")

    def get_encoded_circuits_table(self):
        return self.get_config_value("ENCODED_CIRCUITS_TABLE")

    def get_output_table(self):
        return self.get_config_value("OUTPUT_TABLE")

    def get_define_circuits_proc(self):
        return self.get_config_value("DEFINE_CIRCUITS_PROC")

    def get_run_powerflow_proc(self):
        return self.get_config_value("RUN_POWERFLOW_PROC")

    def get_register_lfe_load_flow_udtf_proc(self):
        return self.get_config_value("REGISTER_LFE_LOAD_FLOW_UDTF_PROC")
    
    def get_run_load_allocation_proc(self):
        return self.get_config_value("RUN_LOAD_ALLOCATION_PROC")    

    def load_config(self, config_filename):
        with open(config_filename, "r") as file:
            file_content = file.read()
            self.config = json.loads(file_content)
            self.env = self.config["active-env"]
            self.stage_name = self.config["stage-name"]

    # def load_config_from_stage(self, session: snowflake.snowpark.Session, stage_name, file_name):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Use the GET command to download the file from the stage
    #         full_stage_path = f"@{stage_name}/{file_name}"
    #         get_command = f"GET {full_stage_path} file://{temp_dir}"
    #         with session.connection.cursor() as cursor:
    #             cursor.execute(get_command)
    #             local_file_path = os.path.join(temp_dir, os.path.basename(file_name))
    #             with open(local_file_path, 'r') as f:
    #                 # data = json.load(f)
    #                 self.config = json.load(f)
    #                 self.env = self.config["active-env"]
    #                 self.stage_name = self.config["stage-name"]

    def load_config_from_json(self, config_json):
        self.config = json.loads(config_json)
        self.env = self.config["active-env"]
        self.stage_name = self.config["stage-name"]

    def load_config_from_parsed_json(self, parsed_config_json):
        self.config = parsed_config_json
        self.env = self.config["active-env"]
        self.stage_name = self.config["stage-name"]

    def get_days_per_batch(self):
        return (
            self.config["days-per-batch"] if "days-per-batch" in self.config else None
        )
        
    def set_circuit_segmenation(self, seg):
        self.config["circuit-segmenation"] = seg
        
    def get_circuit_segmenation(self):
        return (
            self.config["circuit-segmenation"] if "circuit-segmenation" in self.config else {}
        )
    
    def get_load_allocation_table_database(self):
        return (
            self.get_config_value("LOAD_ALLOCATION_TABLE_DATABASE")
            if "LOAD_ALLOCATION_TABLE_DATABASE" in self.get_active_config()
            else self.get_config_value("DATABASE")
        )
        
    def get_load_allocation_table_schema(self):
        return (
            self.get_config_value("LOAD_ALLOCATION_TABLE_SCHEMA")
            if "LOAD_ALLOCATION_TABLE_SCHEMA" in self.get_active_config()
            else self.get_config_value("DATABASE_SCHEMA")
        )
        
    def get_load_allocation_table(self):
        return self.get_config_value("LOAD_ALLOCATION_TABLE")
    
    def get_load_allocation_view_database(self):
        return (
            self.get_config_value("LOAD_ALLOCATION_VIEW_DATABASE")
            if "LOAD_ALLOCATION_VIEW_DATABASE" in self.get_active_config()
            else self.get_config_value("DATABASE")
        )
        
    def get_load_allocation_view_schema(self):
        return (
            self.get_config_value("LOAD_ALLOCATION_VIEW_SCHEMA")
            if "LOAD_ALLOCATION_VIEW_SCHEMA" in self.get_active_config()
            else self.get_config_value("DATABASE_SCHEMA")
        )
        
    def get_load_allocation_view(self):
        return self.get_config_value("LOAD_ALLOCATION_VIEW")