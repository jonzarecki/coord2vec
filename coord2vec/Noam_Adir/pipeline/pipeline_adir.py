from unittest import TestCase
import re

from geopandas import GeoDataFrame
from shapely import wkt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder


def get_csv_data():  # -> Tuple[Tuple[float, float], pd.DataFrame, Any]:

    server_csv_path = "/data/home/morpheus/coord2vec_noam/coord2vec/evaluation/tasks/house_pricing/Housing price in Beijing.csv"
    df = pd.read_csv(server_csv_path, engine='python')
    #     print(df)
    df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)
    coords = df['coord'].values
    features = df[["DOM", "followers", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom",
                   "floor", "buildingType", "constructionTime", "renovationCondition", "buildingStructure",
                   "ladderRatio",
                   "elevator", "fiveYearsProperty", "subway", "district", "communityAverage", "totalPrice"]]
    # in features all csv exept: 'url', 'id', 'Lng', 'Lat', 'coord', "Cid", "tradeTime",
    return coords, features


def generic_clean_col(df, clean_funcs):
    ''' df - data frame
        cols - list of strings contains cols that should be cleaned
        clean_funcs - list of funcs that clean cols that should be cleand in df
    '''
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
    cleaned_df[cleaned_df['constructionTime'].apply(lambda time: not time.isnumeric())] = 0
    return cleaned_df
