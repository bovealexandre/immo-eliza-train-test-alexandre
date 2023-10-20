from utils.prepocessing import preprocessing
from utils.ml import regressor
import pandas as pd

if __name__ == "__main__":
    train = pd.read_json("train.json")
    test = pd.read_json("test.json")

    combined_df = pd.concat([train, test], axis=0, ignore_index=True)

    df = preprocessing(combined_df)

    regressor(df)
