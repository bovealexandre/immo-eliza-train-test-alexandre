import numpy as np
import pandas as pd
from scipy.stats import zscore


def remove_outliers(df, col, threshold):
    z_scores = zscore(df[col])

    outlier_mask = abs(z_scores) > threshold

    df = df[~outlier_mask]
    return df


def price_per_square_meter_per_postal_code(df):
    copied_df = df.copy()

    copied_df["SurfaceOfGood"].fillna(df["LivingArea"], inplace=True)

    pivot_table = (
        copied_df.groupby(["PostalCode"])
        .apply(lambda x: x["Price"].mean() / x["SurfaceOfGood"].mean())
        .reset_index()
    )
    pivot_table.columns = ["PostalCode", "PricePerSquareMeter"]

    pivot_table["PricePerSquareMeter"] = pivot_table["PricePerSquareMeter"].replace(
        np.inf, np.nan
    )

    df = df.merge(pivot_table, on=["PostalCode"], how="left")
    return df


def add_year_of_construction(df):
    current_year = 2023  # You can adjust this to the current year
    df["AgeOfProperty"] = current_year - df["ConstructionYear"]
    return df


def add_price_category(df):
    price_categories = ["Low", "Medium", "High"]
    price_bins = [0, 200000, 500000, np.inf]
    df["PriceCategory"] = pd.cut(df["Price"], bins=price_bins, labels=price_categories)

    return df
