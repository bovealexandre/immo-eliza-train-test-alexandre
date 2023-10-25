import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.func import remove_outliers

# from utils.RandomForestRegressor import regressor
from utils.ml import regressor

# from utils.prep import training_cleaning
from utils.prepocessing import preprocessing

if __name__ == "__main__":
    df = pd.read_json("train.json")

    df = remove_outliers(df, "LivingArea", 3)

    df = preprocessing(df)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor(X_train, X_test, y_train, y_test)
