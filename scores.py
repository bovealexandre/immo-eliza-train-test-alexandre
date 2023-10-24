# import the class of the model you've used
import json
import pickle
import sys

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from utils.prep import test_cleaning

if __name__ == "__main__":
    df = pd.read_json(sys.argv[1])
    print(df.shape)
    print(df.head())

    df_encoded = test_cleaning(df)

    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"].values

    print(X.shape, y.shape)

    model = pickle.load(open("model.pkl", "rb"))
    scores = {"r2": model.score(X, y), "mse": mean_absolute_error(y, model.predict(X))}
    print(scores)
