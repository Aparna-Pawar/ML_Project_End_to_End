import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # --- Imbalance Handling: scale_pos_weight ---
            # Ratio of Negative to Positive classes
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            scale_weight = neg / pos if pos > 0 else 1
            
            logging.info(f"Class Imbalance Info - Neg: {neg}, Pos: {pos}, Scale Weight: {scale_weight:.2f}")

            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
                "XGBoost": XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            # Define wider parameter distributions for RandomizedSearch
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, None],
                    'min_samples_leaf': [1, 2, 4]
                },
                "XGBoost": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5]
                }
            }

            logging.info("Starting evaluation with Recall focus and Threshold Tuning...")
            model_report = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                param=params
            )

            # --- Select Best Model based on RECALL ---
            best_model_name = max(model_report, key=lambda x: model_report[x]['recall'])
            best_model_data = model_report[best_model_name]
            
            best_model_score = best_model_data['recall']
            best_model_obj = best_model_data['model_obj']
            best_threshold = best_model_data['best_threshold']

            logging.info(f"Best Model Found: {best_model_name}")
            logging.info(f"Recall: {best_model_score:.2%}, Optimal Threshold: {best_threshold:.4f}")

            # Safety check: We want at least some predictive power for recall
            if best_model_score < 0.4:
                raise CustomException("Model Recall is too low (below 40%). Need more features or data.", sys)

            # --- Save Model + Threshold ---
            # We save both because prediction MUST use the same custom threshold
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    "model": best_model_obj,
                    "threshold": best_threshold
                }
            )

            # Log final classification report for visibility
            final_probs = best_model_obj.predict_proba(X_test)[:, 1]
            final_preds = (final_probs >= best_threshold).astype(int)
            
            logging.info(f"\nFinal Test Set Results ({best_model_name}):\n{classification_report(y_test, final_preds)}")

            # 4. CREATE AND SAVE CONFUSION MATRIX
            cm = confusion_matrix(y_test, final_preds)
            logging.info(f"\nConfusion Matrix :\n{confusion_matrix(y_test, final_preds)}")


            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)