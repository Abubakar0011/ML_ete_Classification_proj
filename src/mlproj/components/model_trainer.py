import os
import sys
from dataclasses import dataclass
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.utils import save_object_file_path
from src.mlproj.utils import evaluate_classification_models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

# Metrics and model evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score


@dataclass
class ModelTrainerConfiguration:
    model_train_file_path = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfiguration()

    def evaluation_metrics(self, actual_val, predicted_val):
        accuracy = accuracy_score(actual_val, predicted_val)
        f1 = f1_score(actual_val, predicted_val, average='weighted')
        precision = precision_score(actual_val, predicted_val,
                                    average='weighted')
        recall = recall_score(actual_val, predicted_val, average='weighted')
        
        return accuracy, f1, precision, recall
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting the train and test set")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Dictionary of models
            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            # Dictionary of hyperparameters for each model
            param = {
                'Logistic Regression': {'C': [0.1, 1.0, 10.0],
                                        'solver': ['liblinear', 'lbfgs']},

                'Random Forest': {'n_estimators': [10, 50, 100],
                                  'max_depth': [None, 10, 20, 30]},

                'SVM': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]},

                'KNN': {'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']},

                'Decision Tree': {'max_depth': [None, 10, 20, 30],
                                  'criterion': ['gini', 'entropy']},

                'AdaBoost': {'n_estimators': [50, 100, 150],
                             'learning_rate': [0.01, 0.1, 1.0]},

                'Gradient Boosting': {'n_estimators': [50, 100, 150],
                                      'learning_rate': [0.01, 0.1, 0.2],
                                      'max_depth': [3, 5, 7]}
            }

            model_report: dict = evaluate_classification_models(
                X_train, y_train, X_test, y_test, models, param)
            
            # Get best model score and name
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name}")

            # Get best parameters for the best model
            best_params = param[best_model_name]

            mlflow.set_registry_uri(
                "https://dagshub.com/Abubakar0011/"
                "ML_ete_Classification_proj.mlflow"
            )

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme

            # Start MLflow run
            with mlflow.start_run():
                # Predict on the test set
                predicted_qualities = best_model.predict(X_test)

                # Calculate classification metrics
                accuracy, f1, precision, recall = self.evaluation_metrics(
                    y_test, predicted_qualities)

                # Log parameters and metrics
                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                # Log the model to MLflow
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model, "model",
                        registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with acceptable accuracy.")
            logging.info("Best model found and logged.")

            save_object_file_path(
                file_path=self.model_train_config.model_train_file_path,
                obj=best_model)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
