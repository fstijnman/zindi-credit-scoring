# Import libraries
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer


# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
import warnings


def load_data():
    warnings.filterwarnings("ignore")

    train = pd.read_csv("data/Train.csv")
    test = pd.read_csv("data/Test.csv")

    data = pd.concat([train, test]).reset_index(drop=True)

    # Convert the datetime columns appropriately
    date_cols = ["disbursement_date", "due_date"]
    for col in date_cols:
        data[col] = pd.to_datetime(data[col])
        # Extract month, day, and year from the date columns
        data[col + "_month"] = data[col].dt.month
        data[col + "_day"] = data[col].dt.day
        data[col + "_year"] = data[col].dt.year

    # Select all categorical columns from the dataset and label encode them or one hot encode
    cat_cols = data.select_dtypes(include="object").columns
    num_cols = [
        col
        for col in data.select_dtypes(include="number").columns
        if col not in ["target"]
    ]
    logger.info(f"The categorical columns are: {cat_cols}.")
    logger.info("-" * 100)
    logger.info(f"The numerical columns are: {num_cols}")
    logger.info("-" * 100)

    # we are going to one  hot encode the loan type
    data = pd.get_dummies(
        data, columns=["loan_type"], prefix="loan_type", drop_first=False
    )
    # Convert all the columns with prefix loan_type_ to 0/1 instead of False/True
    loan_type_cols = [col for col in data.columns if col.startswith("loan_type_")]
    data[loan_type_cols] = data[loan_type_cols].astype(int)

    # Label-encoding for the other remaining categorical columns
    le = LabelEncoder()
    for col in [col for col in cat_cols if col not in ["loan_type", "ID"]]:
        data[col] = le.fit_transform(data[col])

    # deal with numerical columns: we saw loan amount is  highly right skewed for this we can log transform it
    data["Total_Amount"] = np.log1p(
        data["Total_Amount"]
    )  # study other numerical columns and see if they are skewed as well

    # Splitting the data back into train and test
    train_df = data[data["ID"].isin(train["ID"].unique())]

    test_df = data[data["ID"].isin(test["ID"].unique())]

    # we are also going to drop the country id as we saw we have only one country in train
    features_for_modelling = [
        col
        for col in train_df.columns
        if col not in date_cols + ["ID", "target", "country_id"]
    ]

    # Check if the new datasets have the same rows as train and test datasets
    logger.info(f"The shape of train_df is: {train_df.shape}")
    logger.info(f"The shape of test_df is: {test_df.shape}")
    logger.info(f"The shape of train is: {train.shape}")
    logger.info(f"The shape of test is: {test.shape}")
    logger.info(f"The features for modelling are:\n{features_for_modelling}")

    return train_df, test_df, features_for_modelling
