import pickle

import pandas as pd


def get_csv_data(use_full_dataset=True) -> pd.DataFrame:
    """
    load data to data frame of Beijing house pricing
    Args:
        use_full_dataset: if True load from the full csv, else load from a not full csv for debugging

    Returns: DataFrame from Housing price in Beijing.csv may not be the complete csv

    """
    server_csv_folder_path = "/data/home/morpheus/coord2vec_noam/coord2vec/evaluation/tasks/house_pricing"
    if use_full_dataset:
        csv_path = f"{server_csv_folder_path}/Housing price in Beijing.csv"
    else:
        small_or_medium = "medium"
        csv_path = f"{server_csv_folder_path}/Housing price in Beijing {small_or_medium}.csv"
    df = pd.read_csv(csv_path, engine='python')
    df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)
    features = df[["DOM", "followers", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom",
                   "floor", "buildingType", "constructionTime", "renovationCondition", "buildingStructure",
                   "ladderRatio", "elevator", "fiveYearsProperty", "subway", "district", "communityAverage", "coord",
                   "totalPrice"]]
    # in features all csv exept: 'url', 'id', 'Lng', 'Lat', 'coord', "Cid", "tradeTime",
    return features


def load_data_from_pickel(file_name: str, longitude_name: str, latitude_name: str) -> pd.DataFrame:
    """
    load geographic data from pickle assumes every row has a longitude col and a latitude col
    Args:
        file_name: the pickle file name to load
        longitude_name: the name of longitude column in df
        latitude_name: the name of latitude column in df

    Returns: a data frame that was pickled with added "coord" col

    """
    df = pickle.load(open(file_name, "rb"))
    df["coord"] = df.apply(lambda row: tuple(row[[longitude_name, latitude_name]].values), axis=1)
    return df


def generic_clean_col(df: pd.DataFrame, clean_funcs) -> pd.DataFrame:
    """
    apply functions of df and return new dataframe
    Args:
        df: data frame
        clean_funcs: list of funcs that clean cols that should be cleand in df

    Returns: cleaned_df w

    """
    for i, col in enumerate(clean_funcs):
        df = clean_funcs[i](df)
    cleaned_df = df.fillna(0)
    return cleaned_df


def clean_floor_col(df):
    """
    cleaning function for Beijing house pricing dataset for "floor" col
    Args:
        df: the df loaded from Beijing house pricing dataset

    Returns: DataFrame with cleaned floor col

    """
    # remove data points with no complete data
    cleaned_df = df.copy()
    cleaned_df = cleaned_df[cleaned_df["floor"].apply(lambda floor: len(floor.split())) == 2]
    cleaned_df["floor"] = cleaned_df["floor"].apply(lambda floor: floor.split()[1])
    return cleaned_df


def clean_constructionTime_col(df):
    """
    cleaning function for Beijing house pricing dataset for "constructionTime" col
    Args:
        df: the df loaded from Beijing house pricing dataset

    Returns: DataFrame with cleaned constructionTime col

    """
    cleaned_df = df.copy()
    # cleaned_df['constructionTime'][cleaned_df['constructionTime'].apply(lambda time: not time.isnumeric())] = 0
    cleaned_df.loc[cleaned_df['constructionTime'].apply(lambda time: not time.isnumeric()), 'constructionTime'] = 0
    return cleaned_df


ALL_FILTER_FUNCS_LIST = [clean_floor_col, clean_constructionTime_col]


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
