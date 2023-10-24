import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.func import remove_outliers
from utils.ml import regressor
from utils.prep import training_cleaning

if __name__ == "__main__":
    df = pd.read_json("train.json")

    df = remove_outliers(df, "LivingArea", 2)

    df = training_cleaning(df)

    X = df.drop("Price", axis=1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Price"])
    encoder.fit(y)

    with open("encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = regressor(X_train, X_test, y_train, y_test)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
