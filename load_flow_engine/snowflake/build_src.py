"""
Build the 'src' directory with all modules needed for the Snowflake powerflow.zip package.
Run this before install.py or the lfe_install notebook.
"""
import os
import shutil

SRC_DIR = "src"

# Modules/packages to include in powerflow.zip
COPY_ITEMS = [
    "powerflow_snowflake",
    "powerflow_pipeline",
    "sce",
    "load_flow_engine",
    "lfe_load_flow_wrapper.py",
    "deploy",
]

def build_src():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if os.path.isdir(SRC_DIR):
        shutil.rmtree(SRC_DIR)
    os.makedirs(SRC_DIR)

    for item in COPY_ITEMS:
        src_path = os.path.join(script_dir, item)
        dst_path = os.path.join(SRC_DIR, item)
        if os.path.isdir(src_path):
            shutil.copytree(
                src_path, dst_path,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
            )
        elif os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: {item} not found at {src_path}, skipping")

    print(f"Built '{SRC_DIR}/' with: {os.listdir(SRC_DIR)}")


if __name__ == "__main__":
    build_src()
