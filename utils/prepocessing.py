import numpy as np
import pandas as pd
from pandas import CategoricalDtype

from utils.func import add_year_of_construction


def preprocessing(df):
    kit_cat = CategoricalDtype(
        categories=[
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

    # price_per_square_meter_per_postal_code = price_per_square_meter_per_postal_code(df)
    df = add_year_of_construction(df)
    df["Amenities"] = df[
        ["Openfire", "Terrace", "SwimmingPool", "Furnished", "Kitchen", "Heating"]
    ].sum(axis=1)
    LivingAreaCategory = ["0", "0-50", "50-100", "100-150", "150-200", "200+"]

    living_area_category = pd.CategoricalDtype(
        categories=LivingAreaCategory, ordered=True
    )
    df["LivingAreaCategory"] = pd.cut(
        df["LivingArea"],
        bins=[-np.inf, 0, 50, 100, 150, 200, np.inf],
        labels=["0", "0-50", "50-100", "100-150", "150-200", "200+"],
    )
    df["LivingAreaCategory"] = (
        df["LivingAreaCategory"].astype(living_area_category).cat.codes
    )
    df = df[df["TypeOfSale"] != 2]
    df = df.drop(columns=["Url", "PropertyId", "TypeOfSale", "SubtypeOfProperty"])

    return df
