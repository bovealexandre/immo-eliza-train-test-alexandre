import pandas as pd
from pandas import CategoricalDtype

from utils.func import add_year_of_construction, price_per_square_meter_per_postal_code


def preprocessing(df):
    kit_cat = CategoricalDtype(
        categories=[
            "not installed",
            "usa uninstalled",
            "semi equipped",
            "usa semi equipped",
            "hyper equipped",
            "usa hyper equipped",
            "installed",
            "usa installed",
        ],
        ordered=True,
    )
    df["Kitchen"] = df["Kitchen"].fillna("not installed")
    df["Kitchen"] = df["Kitchen"].astype(kit_cat).cat.codes

    building_state_type = pd.CategoricalDtype(
        categories=[
            "not known",
            "to be done up",
            "to restore",
            "to renovate",
            "just renovated",
            "good",
            "as new",
        ],
        ordered=True,
    )
    df["StateOfBuilding"] = df["StateOfBuilding"].fillna("not known")
    df["StateOfBuilding"] = df["StateOfBuilding"].astype(building_state_type).cat.codes

    heating_cat = CategoricalDtype(
        categories=[
            "not known",
            "fueloil",
            "gas",
            "carbon",
            "wood",
            "pellet",
            "electric",
            "solar",
        ],
        ordered=True,
    )
    df["Heating"] = df["Heating"].fillna("not known")
    df["Heating"] = df["Heating"].astype(heating_cat).cat.codes

    df["SubtypeOfProperty"] = df["SubtypeOfProperty"].fillna("not known")
    one_hot_kit = pd.get_dummies(df["SubtypeOfProperty"], prefix="SubtypeOfProperty")
    df = pd.concat([df, one_hot_kit], axis=1)
    df = df.drop("SubtypeOfProperty", axis=1)

    df = price_per_square_meter_per_postal_code(df)
    df = add_year_of_construction(df)
    df = df.drop(columns=["Url", "PropertyId"])

    return df
