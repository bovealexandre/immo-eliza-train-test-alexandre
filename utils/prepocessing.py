import pandas as pd
from utils.func import remove_outliers, price_per_square_meter_per_postal_code
from pandas import CategoricalDtype


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
    df["SubtypeOfProperty"] = df["SubtypeOfProperty"].astype("category").cat.codes

    df = remove_outliers(df, "Price", 0.05, 0.95)

    df = price_per_square_meter_per_postal_code(df)

    df.dropna(subset=["PricePerSquareMeter"], axis=0)

    df = remove_outliers(df, "PricePerSquareMeter", 0.05, 0.95)

    df["SurfaceOfGood"] = df["SurfaceOfGood"].fillna(df["LivingArea"])
    df = df.drop(
        columns=["Url", "PropertyId", "GardenArea", "ConstructionYear", "Openfire"]
    )

    df = df.dropna()
    return df
