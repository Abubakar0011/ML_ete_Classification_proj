import os
import sys
import pymysql
import pandas as pd
from dotenv import load_dotenv
from src.mlproj.exceptions import CustomException
from src.mlproj.logger import logging

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
Database = os.getenv('Database')

''' To connect sql database'''


def reading_SQL_data():
    logging.info("Reading SQL database has started.")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=Database
        )
        logging.info("Connection with Database established", mydb)
        df = pd.read_sql_query('select * from customers', mydb)
        print(df.head())

        return df
    except Exception as e:
        raise CustomException(e, sys)










