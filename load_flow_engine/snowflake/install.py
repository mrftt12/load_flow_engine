import pathlib
from deploy.snowflake_procedure_runner import SnowflakeSProcClient
from powerflow_pipeline.util import PowerflowConfig
from dotenv import load_dotenv

from powerflow_snowflake.procedure import define_circuit, register_lfe_load_flow_udtf, run_powerflow

script_dir = pathlib.Path(__file__).parent.resolve()
load_dotenv()


def main():
    pf_config = PowerflowConfig()
    current_dir = pathlib.Path(__file__).parent.resolve()
    pf_config.load_config(f"{current_dir}/env.json")
    cli = SnowflakeSProcClient(
        stage_name=f"{pf_config.get_database()}.{pf_config.get_database_schema()}.{pf_config.get_stage_name()}",
        pkg_name="powerflow",
        pkg_src="src",
        recreate_stage=True,
        local_staging_dir="_staging",
    )
    procs = {
        f"{pf_config.get_database()}.{pf_config.get_database_schema()}.{pf_config.get_define_circuits_proc()}": define_circuit,
        f"{pf_config.get_database()}.{pf_config.get_database_schema()}.{pf_config.get_run_powerflow_proc()}": run_powerflow,
        f"{pf_config.get_database()}.{pf_config.get_database_schema()}.{pf_config.get_register_lfe_load_flow_udtf_proc()}": register_lfe_load_flow_udtf,
    }
    cli.install(procs)


if __name__ == "__main__":
    main()
