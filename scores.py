# import the class of the model you've used
from catboost import CatBoostRegressor
import sys, json, pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error
from utils.prepocessing import preprocessing

if __name__ == "__main__":
    df = pd.read_json(sys.argv[1])
    print(df.shape)
    print(df.head())

    df_encoded = preprocessing(df)

    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"].values

    print(X.shape, y.shape)

    model = pickle.load(open("best_catboost_model.pkl", "rb"))
    scores = {"r2": model.score(X, y), "mse": mean_absolute_error(y, model.predict(X))}
    print(scores)
