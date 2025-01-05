# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def create_binary_classifier(X, y, class_value=1, sampling_method=None, model_type='random_forest',
                             sampling_strategy='minority', threshold=0.5, class_weight='balanced'):
    """
    This function creates a binary classifier for a given X and y dataset for a specific class.

    Parameters:
    - X: Features
    - y: Labels
    - class_value: The value of the target class (e.g., 1 or 0)
    - sampling_method: Sampling technique ('random', 'smote', 'adasyn', 'under_sampling', or None)
    - model_type: The model to be used ('random_forest', 'gradient_boosting', 'lightgbm', 'logistic_regression', 'xgboost')
    - sampling_strategy: Sampling strategy ('minority' or 'majority')
    - threshold: Decision threshold (default is 0.5, but can be adjusted based on model output)
    - class_weight: Class weights ('balanced' or None)

    Returns:
    - clf: The trained classifier
    """
    # Separate the minority class (class_value) and others
    X_class = X[y == class_value]
    y_class = y[y == class_value]
    X_others = X[y != class_value]
    y_others = y[y != class_value]

    # Combine the minority class (class_value) with the others
    X_combined = pd.concat([X_class, X_others], axis=0)
    y_combined = pd.concat([y_class, y_others], axis=0)

    # Apply sampling if method is specified
    if sampling_method:
        if sampling_method == 'random':
            ros = RandomOverSampler(sampling_strategy='auto')  # Auto-sampling
            X_res, y_res = ros.fit_resample(X_combined, y_combined)
        elif sampling_method == 'smote':
            smote = SMOTE(sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_combined, y_combined)
        elif sampling_method == 'adasyn':
            adasyn = ADASYN(sampling_strategy='auto')
            X_res, y_res = adasyn.fit_resample(X_combined, y_combined)
        elif sampling_method == 'under_sampling':
            rus = RandomUnderSampler(sampling_strategy='auto')
            X_res, y_res = rus.fit_resample(X_combined, y_combined)
        elif sampling_method == 'none':
            X_res, y_res = X_combined, y_combined
        else:
            raise ValueError("Invalid sampling_method. Options are 'random', 'smote', 'adasyn', 'under_sampling' or 'none'.")
    else:
        X_res, y_res = X_combined, y_combined

    # Choose the model based on input
    if model_type == 'random_forest':
        clf = RandomForestClassifier(class_weight=class_weight)
    elif model_type == 'gradient_boosting':
        clf = GradientBoostingClassifier()
    elif model_type == 'lightgbm':
        clf = lgb.LGBMClassifier()
    elif model_type == 'logistic_regression':
        clf = LogisticRegression(class_weight=class_weight)
    elif model_type == 'xgboost':
        clf = XGBClassifier(scale_pos_weight=1.0 if class_weight == 'balanced' else class_weight)
    else:
        raise ValueError("Invalid model_type. Options are 'random_forest', 'gradient_boosting', 'lightgbm', 'logistic_regression', 'xgboost'.")

    # Train the model
    clf.fit(X_res, y_res)

    # If threshold is set, adjust the decision boundary
    if threshold != 0.5:
        probas = clf.predict_proba(X_res)
        y_pred = (probas[:, 1] >= threshold).astype(int)
        return clf, y_pred
    else:
        return clf


def train_other_classes(X, y, class_value=1, model_type='random_forest', class_weight='balanced'):
    """
    This function trains a classifier for all classes except the specified class_value.

    Parameters:
    - X: Features
    - y: Labels
    - class_value: The class to exclude
    - model_type: The model to use ('random_forest', 'gradient_boosting', 'lightgbm', 'logistic_regression', 'xgboost')
    - class_weight: Class weights ('balanced' or None)

    Returns:
    - clf_others: The trained classifier for other classes
    """
    X_class0_others = X[y != class_value]
    y_class0_others = y[y != class_value]

    if class_weight == 'balanced':
        class_weights = compute_sample_weight(class_weight='balanced', y=y_class0_others)
    else:
        class_weights = None

    if model_type == 'random_forest':
        clf_others = RandomForestClassifier(class_weight=class_weight)
    elif model_type == 'gradient_boosting':
        clf_others = GradientBoostingClassifier()
    elif model_type == 'lightgbm':
        clf_others = lgb.LGBMClassifier()
    elif model_type == 'logistic_regression':
        clf_others = LogisticRegression(class_weight=class_weight)
    elif model_type == 'xgboost':
        pos_class_count = np.sum(y_class0_others == 1)
        neg_class_count = np.sum(y_class0_others == 0)
        scale_pos_weight = 1 if pos_class_count == 0 or neg_class_count == 0 else float(neg_class_count / pos_class_count)
        clf_others = XGBClassifier(scale_pos_weight=scale_pos_weight)
    
    if model_type in ['random_forest', 'logistic_regression']:
        clf_others.fit(X_class0_others, y_class0_others, sample_weight=class_weights)
    else:
        clf_others.fit(X_class0_others, y_class0_others)

    return clf_others


class FinalClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that combines the results from two different classifiers.
    """

    def __init__(self, clf_class1=None, clf_others=None):
        self.clf_class1 = clf_class1
        self.clf_others = clf_others

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)

        self.clf_class1_ = clone(self.clf_class1)
        self.clf_others_ = clone(self.clf_others)

        if hasattr(self.clf_class1_, "random_state"):
            self.clf_class1_.set_params(random_state=42)
        if hasattr(self.clf_others_, "random_state"):
            self.clf_others_.set_params(random_state=42)

        self.clf_class1_.fit(X, y)
        self.clf_others_.fit(X, y)

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, ['clf_class1_', 'clf_others_', 'classes_'])

        X = check_array(X)

        pred1 = self.clf_class1_.predict(X)
        pred2 = self.clf_others_.predict(X)

        final_prediction = []
        for p1, p2 in zip(pred1, pred2):
            final_prediction.append(p1 if p1 == p2 else p1)  # Choose p1 or p2 as needed

        return np.array(final_prediction)
        
    def predict_proba(self, X):
        check_is_fitted(self, ['clf_class1_', 'clf_others_', 'classes_'])
        X = check_array(X)

        prob1 = self.clf_class1_.predict_proba(X)
        prob2 = self.clf_others_.predict_proba(X)

        return np.concatenate([prob1, prob2], axis=1)