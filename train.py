from loguru import logger
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt


def train_model(train_df, features_for_modelling, params):

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df[features_for_modelling],
        train_df["target"],
        stratify=train_df["target"],
        shuffle=True,
        random_state=42,
    )
    logger.info(
        f"Training with {X_train.shape[0]} rows and validating with {X_valid.shape[0]} rows"
    )

    oof_predictions = np.zeros(len(train_df))
    feature_importance_df = pd.DataFrame()
    f1_scores = []
    roc_auc_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df[features_for_modelling], train_df["target"])
    ):
        logger.info(f"\nFold {fold + 1}")

        X_train = train_df[features_for_modelling].iloc[train_idx]
        X_valid = train_df[features_for_modelling].iloc[valid_idx]
        y_train = train_df["target"].iloc[train_idx]
        y_valid = train_df["target"].iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=5000,
        )

        y_pred_proba = model.predict(X_valid)
        y_pred = (y_pred_proba > 0.5).astype(int)
        oof_predictions[valid_idx] = y_pred_proba

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features_for_modelling
        fold_importance["importance"] = model.feature_importance()
        fold_importance["fold"] = fold
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance], axis=0
        )

        fold_f1 = f1_score(y_valid, y_pred)
        fold_roc_auc = roc_auc_score(y_valid, y_pred_proba)
        f1_scores.append(fold_f1)
        roc_auc_scores.append(fold_roc_auc)

        logger.info(f"F1 Score: {fold_f1:.4f}")
        logger.info(f"ROC AUC Score: {fold_roc_auc:.4f}")
        logger.info(
            f"\nClassification Report:\n {classification_report(y_valid, y_pred)}"
        )

    plt.figure(figsize=(10, 6))
    feature_importance_df.groupby("feature")["importance"].mean().sort_values(
        ascending=False
    ).head(20).plot(kind="barh")
    plt.title("Average Feature Importance")
    plt.show()

    logger.info("Overall Performance:")
    logger.info(f"Average F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    logger.info(
        f"Average ROC AUC: {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}"
    )

    return oof_predictions
