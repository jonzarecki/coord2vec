import pandas as pd


def get_csv_data(use_full_dataset=True) -> pd.DataFrame:
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


def generic_clean_col(df, clean_funcs):
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
    # remove data points with no complete data
    cleaned_df = df.copy()
    cleaned_df = cleaned_df[cleaned_df["floor"].apply(lambda floor: len(floor.split())) == 2]
    cleaned_df["floor"] = cleaned_df["floor"].apply(lambda floor: floor.split()[1])
    return cleaned_df


def clean_constructionTime_col(df):
    cleaned_df = df.copy()
    # cleaned_df['constructionTime'][cleaned_df['constructionTime'].apply(lambda time: not time.isnumeric())] = 0
    cleaned_df.loc[cleaned_df['constructionTime'].apply(lambda time: not time.isnumeric()), 'constructionTime'] = 0
    return cleaned_df


ALL_FILTER_FUNCS_LIST = [clean_floor_col, clean_constructionTime_col]
