def parse_sold(sold: str) -> float:
    """
    parsing method for manhattan dataset sold column
    Args:
        sold: the representation in the dataset

    Returns: float representation of the price in sold column

    """
    sold = sold.lower()
    if "m" in sold:
        return float(sold[:sold.index("m")]) * 1000000
    return float(sold)


def clean_manhattan_sold_col(df):
    """
    cleaning function for manhattan house pricing dataset for "sold" col
    Args:
        df: the df from manhattan house pricing dataset

    Returns: DataFrame with cleaned sold col

    """
    cleaned_df = df.copy()
    cleaned_df["sold"] = cleaned_df.apply(lambda row: parse_sold(row["sold"]), axis=1)
    return cleaned_df


ALL_MANHATTAN_FILTER_FUNCS_LIST = [clean_manhattan_sold_col]




