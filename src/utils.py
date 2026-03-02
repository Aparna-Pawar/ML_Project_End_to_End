import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            # --- Randomized Search CV ---
            # n_iter=10 tries 10 random combinations; scoring='recall' focuses on fraud detection
            rs = RandomizedSearchCV(
                model, 
                param_distributions=para, 
                n_iter=15, 
                cv=skf, 
                verbose=1, 
                random_state=42, 
                n_jobs=-1, 
                scoring='recall'
            )
            rs.fit(X_train, y_train)

            # Update model with best parameters found
            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            # --- Threshold Tuning Logic ---
            # Get probabilities for the positive class (Fraud)
            y_probs = model.predict_proba(X_test)[:, 1]
            
            # Calculate precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
            
            # Find threshold that maximizes F1-score (balancing Recall and Precision)
            # 1e-10 prevents division by zero
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            
            # If no threshold found (rare), default to 0.5
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


            # Apply the optimized threshold
            y_pred_tuned = (y_probs >= best_threshold).astype(int)

            # Store the comprehensive report
            report[model_name] = {
                "recall": recall_score(y_test, y_pred_tuned),
                "precision": precision_score(y_test, y_pred_tuned, zero_division=0),
                "f1": f1_score(y_test, y_pred_tuned),
                "best_threshold": best_threshold,
                "model_obj": model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def model_metrics(true, predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
    

def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)