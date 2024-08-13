import os
import sys
import pymysql
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
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


'''It is ageneric routine will take the preprocessor object file path and
will dump that prpprocessor into pickle file'''


def save_object_file_path(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification_models(X_train, y_train, X_test,
                                   y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Use accuracy as scoring
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics for training data
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            train_precision = precision_score(y_train, y_train_pred,
                                              average='weighted')
            train_recall = recall_score(y_train, y_train_pred,
                                        average='weighted')

            logging.info(
                f"Model: {model}, Train Accuracy: {train_accuracy:.4f}, "
                f"Train F1 Score: {train_f1:.4f}"
            )
            logging.info(
                f"Model: {model}, Train Precision: {train_precision:.4f}, "
                f"Train Recall: {train_recall:.4f}"
            )

            # Calculate metrics for test data
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred,
                                             average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')

            # Add the test accuracy to the report
            report[list(models.keys())[i]] = {
                'Test Accuracy': test_accuracy,
                'Test F1 Score': test_f1,
                'Test Precision': test_precision,
                'Test Recall': test_recall,
                'Confusion Matrix': confusion_matrix(y_test, y_test_pred)
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


