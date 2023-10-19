def remove_outliers(df, col, min, max):
    Q1 = df[col].quantile(min)
    Q3 = df[col].quantile(max)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]
    return df
