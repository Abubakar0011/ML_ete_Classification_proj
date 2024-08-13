from src.mlproj.exceptions import CustomException
from src.mlproj.logger import logging
import sys
from src.mlproj.components.data_ingestion import DataIngestion
from src.mlproj.components.data_transformation import DataTransformation
from src.mlproj.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    logging.info("Execution has started")
    try:
        dataIngestion = DataIngestion()
        trn_dta_pat, tst_dta_pat = dataIngestion.data_ingestion_initiation()

        data_tarnsf = DataTransformation()
        train_arr, test_arr = data_tarnsf.initiate_data_transformation(
            trn_dta_pat, tst_dta_pat
        )
        model_train = ModelTrainer()
        print(model_train.initiate_model_training(train_arr, test_arr))

    except Exception as e:
        raise CustomException(e, sys)





