from etl import load_data
from train import train_model

if __name__ == "__main__":
    train_df, test_df, features_for_modelling = load_data()
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "early_stopping_rounds": 100,
        "is_unbalance": True,
    }

    train_model(train_df, features_for_modelling, params)
