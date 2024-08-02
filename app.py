from src.mlproj.exceptions import CustomException
from src.mlproj.logger import logging
import sys
from src.mlproj.components.data_ingestion import DataIngestion


if __name__ == '__main__':
    logging.info("Execution has started")
    try:
        dataIngestion = DataIngestion()
        dataIngestion.data_ingestion_initiation()
    except Exception as e:
        raise CustomException(e, sys)
