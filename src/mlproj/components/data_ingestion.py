import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.mlproj.exceptions import CustomException
from src.mlproj.logger import logging
from dataclasses import dataclass
# from src.mlproj.utils import reading_SQL_data


@dataclass
class DataIngestionConfiguration:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfiguration()

    '''This method read data from sql db and will return the train
    and test path'''

    def data_ingestion_initiation(self):
        try:
            # df = reading_SQL_data()
            # logging.info("Reading data from SQL DB.")

            df = pd.read_csv(os.path.join('notebook/Data', 'raw.csv'))

            os.makedirs(os.path.dirname(
                self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,
                      index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2,
                                                   random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,
                             index=False, header=True)

            test_set.to_csv(self.data_ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Data ingestion has been completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
