import numpy as np


def remove_outliers(df, col, min, max):
    Q1 = df[col].quantile(min)
    Q3 = df[col].quantile(max)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def price_per_square_meter_per_postal_code(df):
    copied_df = df.copy()

    copied_df["SurfaceOfGood"].fillna(0, inplace=True)
    copied_df["LivingArea"].fillna(0, inplace=True)

    pivot_table = (
        copied_df.groupby("PostalCode")
        .apply(
            lambda x: x["Price"].mean() / (x["SurfaceOfGood"] + x["LivingArea"]).mean()
        )
        .reset_index()
    )
    pivot_table.columns = ["PostalCode", "PricePerSquareMeter"]

    pivot_table["PricePerSquareMeter"] = pivot_table["PricePerSquareMeter"].replace(
        np.inf, np.nan
    )

    df = df.merge(pivot_table, on="PostalCode", how="left")
    return df
