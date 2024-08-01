import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name = 'mlproj'

file_list = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/predictions_pipeline.py",
    f"src/{project_name}/exceptions.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "app.py",
    "setup.py"
]

for fl_path in file_list:
    file_path = Path(fl_path)
    fdir, fname = os.path.split(file_path)

    if fdir != "":
        os.makedirs(fdir, exist_ok=True)
        logging.info(f"Creating directory: {fdir} for the file {fname}.")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w'):
            logging.info(f"Creating an empty file: {file_path}")
    else:
        logging.info(f"{fname} already exists.")