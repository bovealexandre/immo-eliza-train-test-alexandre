import pandas as pd


def category(df, column):
    """
    Encode a specific column in a pandas DataFrame as a categorical variable using numerical codes.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame.
    - column (str): The name of the column to be encoded.

    Returns:
    - pd.DataFrame: A modified DataFrame with the specified column encoded as categorical.
    """
    df[column] = df[column].astype("category").cat.codes
    return df


def category_state(df):
    """
    Encode the 'StateOfBuilding' column in a pandas DataFrame as an ordered categorical variable using numerical codes.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame.

    Returns:
    - pd.DataFrame: A modified DataFrame with the 'StateOfBuilding' column encoded as an ordered categorical variable.
    """
    building_state_type = pd.CategoricalDtype(
        categories=[
            "to be done up",
            "to restore",
            "to renovate",
            "good",
            "just renovated",
            "as new",
        ],
        ordered=True,
    )
    df["StateOfBuilding"] = df["StateOfBuilding"].astype(building_state_type).cat.codes
    return df


def category_kitchen(df):
    """
    Encode the 'Kitchen' column in a pandas DataFrame as an ordered categorical variable using numerical codes.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame containing a 'Kitchen' column.

    Returns:
    - pd.DataFrame: A modified DataFrame with the 'Kitchen' column encoded as an ordered categorical variable.
    """
    kitchen_type = pd.CategoricalDtype(
        categories=[
            "installed",
            "semi equipped",
            "usa installed",
            "usa semi equipped",
            "hyper equipped",
            "usa hyper equipped",
        ],
        ordered=True,
    )
    df["Kitchen"] = df["Kitchen"].astype(kitchen_type).cat.codes
    return df


def new_features(df):
    """
    Create new features in a pandas DataFrame based on mean values by postal code.

    This function calculates the mean price and mean living area for each postal code
    and creates new features such as 'Price_mean/PostalCode' and 'LivingArea_mean/PostalCode'.
    It also computes the 'Price/SQMeter/PostalCode' feature by dividing the mean price by the mean living area.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame with real estate data.

    Returns:
    - pd.DataFrame: A DataFrame with the added new features.
    """
    # Calculate mean price by postal code
    mean_price_by_postal = df.groupby("PostalCode")["Price"].mean().reset_index()
    # Calculate mean living area by postal code
    mean_living_area_by_postal = (
        df.groupby("PostalCode")["LivingArea"].mean().reset_index()
    )

    # Merge mean price and mean living area back into the DataFrame
    df = df.merge(
        mean_price_by_postal, on="PostalCode", suffixes=("", "_mean/PostalCode")
    )
    df = df.merge(
        mean_living_area_by_postal, on="PostalCode", suffixes=("", "_mean/PostalCode")
    )

    # Calculate 'Price/SQMeter/PostalCode' by dividing mean price by mean living area
    df["Price/SQMeter/PostalCode"] = (
        df["Price_mean/PostalCode"] / df["LivingArea_mean/PostalCode"]
    )
    return df


def drop_outliers_iqr(df, column):
    percentile25_price = df[column].quantile(0.25)
    percentile75_price = df[column].quantile(0.75)
    upper_limit_price = percentile75_price + 1.5 * percentile75_price
    lower_limit_price = percentile25_price - 1.5 * percentile25_price
    df = df[df[column] >= lower_limit_price]
    df = df[df[column] <= upper_limit_price]

    return df


def drop_outliers_z(df, column):
    upper_limit = df[column].mean() + 3 * df[column].std()
    lower_limit = df[column].mean() - 3 * df[column].std()
    df = df[df[column] >= lower_limit]
    df = df[df[column] <= upper_limit]

    return df


def drop_less_5(df, column):
    count = df[column].value_counts()
    keep = count[count >= 3].index
    df = df[df[column].isin(keep)].reset_index()

    return df


def training_cleaning(df):
    """
    Perform preprocessing on a pandas DataFrame containing the immo eliza real estate data.

    This function executes various data preprocessing steps, including handling missing values,
    encoding categorical variables, removing outliers, and creating new features.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame with immo eliza real estate data.

    Returns:
    - pd.DataFrame: A preprocessed DataFrame with cleaned and transformed data.
    """

    df.drop_duplicates()

    # Encode 'SubtypeOfProperty' and 'Heating' as categorical variables
    df["SubtypeOfProperty"].fillna(value="not_known", inplace=True)
    df = category(df, "SubtypeOfProperty")

    df["Heating"].fillna(value="not_known", inplace=True)
    df = category(df, "Heating")

    # Encode 'StateOfBuilding' as an ordered categorical variable
    df["StateOfBuilding"].fillna(value="not_known", inplace=True)
    df = category_state(df)

    # Encode 'Kitchen' as an ordered categorical variable
    df["Kitchen"].fillna(value="not_known", inplace=True)
    df = category_kitchen(df)

    # Handle missing values in 'SurfaceOfGood' based on property type
    apart = df[df["TypeOfProperty"] != 1]
    df["SurfaceOfGood"].fillna(value=apart["LivingArea"])

    df = new_features(df)

    # Drop unnecessary columns
    df.drop(["Url", "PropertyId", "LivingArea_mean/PostalCode"], axis=1, inplace=True)

    df = df.fillna(-1)

    print("Preprocessing Done")
    print(df.shape)
    print(df.columns)

    return df


def test_cleaning(df):
    """
    Perform preprocessing on a pandas DataFrame containing the immo eliza real estate data.

    This function executes various data preprocessing steps, including handling missing values,
    encoding categorical variables, removing outliers, and creating new features.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame with immo eliza real estate data.

    Returns:
    - pd.DataFrame: A preprocessed DataFrame with cleaned and transformed data.
    """
    df.dropna(subset="Price", inplace=True)

    # Encode 'SubtypeOfProperty' and 'Heating' as categorical variables
    df["SubtypeOfProperty"].fillna(value="not_known", inplace=True)
    df = category(df, "SubtypeOfProperty")

    df["Heating"].fillna(value="not_known", inplace=True)
    df = category(df, "Heating")

    # Encode 'StateOfBuilding' as an ordered categorical variable
    df["StateOfBuilding"].fillna(value="not_known", inplace=True)
    df = category_state(df)

    # Encode 'Kitchen' as an ordered categorical variable
    df["Kitchen"].fillna(value="not_known", inplace=True)
    df = category_kitchen(df)

    # Handle missing values in 'SurfaceOfGood' based on property type
    apart = df[df["TypeOfProperty"] != 1]
    df["SurfaceOfGood"].fillna(value=apart["LivingArea"], inplace=True)

    df = new_features(df)

    df.reset_index()

    # Drop unnecessary columns
    df.drop(["Url", "PropertyId"], axis=1, inplace=True)

    print("Preprocessing Done")
    print(df.shape)
    print(df.columns)

    return df
