# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import functools

# Sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# MLFlow
from src.tracking import mlflow_logger
from mlflow import log_param, log_metric
import mlflow.sklearn
from mlflow.models.signature import infer_signature




class ModelEvaluation:
    """
    Supports the evaluation of classification models (multinomial), collecting the results.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, tag: str, test_size: float = 0.2, shuffle: bool = True, random_state: int = 42):
        """
        :param X: the inputs
        :param y: the prediction targets
        :param test_size: the fraction of the data to reserve for testing
        :param shuffle: whether to shuffle the data prior to splitting
        :param random_state: the random seed
        :param tag: target name for logging
        """

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y,
            random_state=random_state, test_size=test_size, shuffle=shuffle)
        
        self.tag = tag

    @mlflow_logger
    def evaluate_model(self, model) -> dict:
        """
        :param model: the model to evaluate
        :return: dict with evaluation metrics
        """
        model_type = type(model.named_steps['model']).__name__
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_valid)

        has_proba = hasattr(model.named_steps['model'], "predict_proba")
        y_score = model.predict_proba(self.X_valid)[:, 1] if has_proba else model.decision_function(self.X_valid)

        metrics = {
            "accuracy":  round(accuracy_score(self.y_valid, y_pred), 3),
            "roc_auc":   round(roc_auc_score(self.y_valid, y_score), 3),
            "f1":        round(f1_score(self.y_valid, y_pred), 3),
            "precision": round(precision_score(self.y_valid, y_pred), 3),
            "recall":    round(recall_score(self.y_valid, y_pred), 3),
        }
        if has_proba:
            metrics["average_precision"] = round(average_precision_score(self.y_valid, y_score), 3)

        print(f"\n{model_type} validation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")

        # Log to MLFlow
        model_name = model_type + '_' + self.tag
        log_param("Model Type", model_name)
        for hyperparameter, value in model.get_params().items():
            log_param(hyperparameter, value)
        for metric, value in metrics.items():
            log_metric(metric, value)

        # Signature
        X_sig = self.X_train.copy()
        int_cols = X_sig.select_dtypes(include=["int64", "int32"]).columns
        if len(int_cols) > 0:
            X_sig[int_cols] = X_sig[int_cols].astype("float64")
        signature = infer_signature(X_sig, model.predict(X_sig))
        mlflow.sklearn.log_model(model, name=model_name, signature=signature, registered_model_name=None)

        return metrics

    def evaluate_cv(self, model, n_splits: int = 5) -> dict:
        """
        :param model: the model to evaluate
        :param n_splits: number of CV folds
        :return: dict with mean CV scores per metric
        """
        model_type = type(model.named_steps['model']).__name__
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # average_precision requires predict_proba — skip for SVM without probability=True
        has_proba = hasattr(model.named_steps['model'], "predict_proba")
        scorings = ["roc_auc", "f1", "precision", "recall"] + (["average_precision"] if has_proba else [])

        results = {"model": model_type}
        for scoring in scorings:
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring, error_score="raise")
            results[scoring] = round(scores.mean(), 3)

        print(f"\n{model_type} CV results ({n_splits}-fold):")
        for k, v in results.items():
            if k != "model":
                print(f"  {k}: {v:.3f}")

        return results