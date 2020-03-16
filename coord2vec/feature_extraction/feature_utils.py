import datetime
import logging
from typing import List, Tuple
import numpy as np
import geopandas as gpd
import pandas as pd
from geoalchemy2 import WKTElement
from shapely import wkt
from shapely.geometry import Polygon, Point
from sklearn.base import TransformerMixin
from sqlalchemy import VARCHAR
from tqdm import tqdm
import socket
from contextlib import closing
from coord2vec.common.db.connectors import get_connection
from coord2vec.common.db.sqlalchemy_utils import get_df, merge_to_table, add_sdo_geo_to_table, insert_into_table, \
    get_temp_table_name
from coord2vec.common.geographic.visualization_utils import get_image_overlay
from coord2vec.config import STEP_SIZE, ors_server_ip, ors_server_port
from coord2vec.feature_extraction.feature_table import FEATURE_NAME, GEOM, GEOM_WKT, FEATURE_VALUE, \
    MODIFICATION_DATE, DTYPES, GEOM_WKT_HASH

# TODO: re-order file, too long

def load_features_using_geoms(input_gs: gpd.GeoSeries, features_table: str,
                              feature_names: List[str] = None) -> gpd.GeoDataFrame:
    """
    Args:
        input_gs: A geo series with geometries to load features on
        features_table: the cache table to load features from
        feature_names: optional. load only a set of features. if None, will load all the features in the table

    Returns:
        A GeoDataFrame with feature names as columns, and input_gs as samples.
        The geometry column in the gdf is GEOM_WKT
    """
    features_table = features_table.lower()
    # create temporary hash table
    input_wkt = input_gs.apply(lambda geo: geo.wkt)
    input_hash = [str(h) for h in pd.util.hash_pandas_object(input_wkt)]
    input_hash_df = pd.DataFrame({GEOM_WKT_HASH: input_hash})

    eng = get_connection(db_name='POSTGRES')
    if not eng.has_table(features_table):  # cache table does not exist
        return gpd.GeoDataFrame()  # just an empty gdf

    tmp_tbl_name = get_temp_table_name()
    insert_into_table(eng, input_hash_df, tmp_tbl_name, dtypes={GEOM_WKT_HASH: VARCHAR(300)})

    add_q = lambda l: ["'" + s + "'" for s in l]
    feature_filter_sql = f"WHERE {FEATURE_NAME} in ({', '.join(add_q(feature_names))})" if feature_names is not None else ""
    # extract the features
    query = f"""
            select {FEATURE_NAME}, {FEATURE_VALUE}, {GEOM_WKT}, f.{GEOM_WKT_HASH}
            from {features_table} f
            join {tmp_tbl_name} t
            on t.{GEOM_WKT_HASH} = f.{GEOM_WKT_HASH}
            {feature_filter_sql}
            """

    results_df = get_df(query, eng)
    pivot_results_df = _pivot_table(results_df)

    # create the full results df
    full_df = pd.DataFrame(data={GEOM_WKT: input_gs.tolist()}, index=input_hash, columns=pivot_results_df.columns)
    assert pivot_results_df.index.isin(full_df.index).all(), "all loaded features should be from the input (according to hash)"
    if not pivot_results_df.empty:
        full_df[full_df.index.isin(pivot_results_df.index)] = pivot_results_df[
            pivot_results_df.index.isin(full_df.index)].values
    full_gdf = gpd.GeoDataFrame(full_df, geometry=GEOM_WKT)
    full_gdf = full_gdf.astype({c: float for c in pivot_results_df.columns if c != GEOM_WKT})

    with eng.begin() as con:
        con.execute(f"DROP TABLE {tmp_tbl_name}")

    return full_gdf


def load_features_in_polygon(polygon: Polygon, features_table: str,
                             feature_names: List[str] = None) -> gpd.GeoDataFrame:
    """
    Extract all the features already calculated inside a polygon
    Args:
        :param polygon: a polygon to get features of all the geometries inside of it
        :param features_table: the name of the features table in which the cache is saved
        :param feature_names: The names of all the features you want to extract. if None, extract all features

    Returns:
        A GeoDataFrame with a features as columns
    """
    features_table = features_table.lower()
    eng = get_connection(db_name='POSTGRES')
    if eng.has_table(features_table):
        # extract data
        add_q = lambda l: ["'" + s + "'" for s in l]
        feature_filter_sql = f"and {FEATURE_NAME} in ({', '.join(add_q(feature_names))})" if feature_names is not None else ""
        query = f"""
                select {FEATURE_NAME}, {FEATURE_VALUE}, CAST(ST_AsText({GEOM}) as TEXT) as {GEOM_WKT}, {GEOM_WKT_HASH}
                from {features_table}
                where ST_Covers(ST_GeomFromText('{polygon.wkt}', 4326)::geography, {GEOM})
                {feature_filter_sql}
                """
        res_df = get_df(query, eng)
        ret_df = _pivot_table(res_df)  # rearrange the df
    else:
        ret_df = gpd.GeoDataFrame()

    eng.dispose()
    return ret_df


def load_all_features(features_table: str) -> gpd.GeoDataFrame:
    """
    Load all the features from the features table in the oracle db
    Returns:
        A Geo Dataframe with all the features as columns
    """
    features_table = features_table.lower()
    eng = get_connection(db_name='POSTGRES')
    query = f"""
            select {FEATURE_NAME}, {FEATURE_VALUE}, CAST(ST_AsText({GEOM}) as TEXT) as {GEOM_WKT}
            from {features_table}
            """
    res_df = pd.read_sql(query, eng)

    eng.dispose()
    return _pivot_table(res_df)


def _pivot_table(df: pd.DataFrame) -> gpd.GeoDataFrame:
    hash2wkt = df[[GEOM_WKT_HASH, GEOM_WKT]].set_index(GEOM_WKT_HASH).to_dict()[GEOM_WKT]
    features_df = df.pivot(index=GEOM_WKT_HASH, columns=FEATURE_NAME,
                           values=FEATURE_VALUE)
    features_df[GEOM_WKT] = [wkt.loads(hash2wkt[h]) for h in features_df.index]
    features_gdf = gpd.GeoDataFrame(features_df, geometry=GEOM_WKT, index=features_df.index)
    return features_gdf


def save_features_to_db(gs: gpd.GeoSeries, df: pd.DataFrame, table_name: str):
    """
    Insert features into the Oracle DB
    Args:
        gs: The geometries of the features
        df: the features, with columns as feature names
        table_name: the features table name in oracle db
    Returns:
        None
    """
    if len(gs) == 0:
        return

    table_name = table_name.lower()
    eng = get_connection(db_name='POSTGRES')

    for column in tqdm(df.columns, desc=f'Inserting Features to {table_name}', unit='feature', leave=False):
        insert_df = pd.DataFrame(data={MODIFICATION_DATE: datetime.datetime.now(),
                                       GEOM: gs.values,
                                       FEATURE_NAME: column,
                                       FEATURE_VALUE: df[column]})
        insert_df[GEOM_WKT] = insert_df[GEOM].apply(lambda g: g.wkt)
        # add hash column for the GEOM_WKT
        insert_df[GEOM_WKT_HASH] = [str(h) for h in pd.util.hash_pandas_object(insert_df[GEOM_WKT])]

        insert_df[GEOM] = insert_df[GEOM].apply(lambda x: WKTElement(x.wkt, srid=4326))
        merge_to_table(eng, insert_df, table_name, compare_columns=[GEOM_WKT_HASH, FEATURE_NAME],
                       update_columns=[MODIFICATION_DATE, FEATURE_VALUE, GEOM, GEOM_WKT], dtypes=DTYPES)
    eng.dispose()


def extract_feature_image(polygon: Polygon, features_table: str, step=STEP_SIZE,
                          feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    features_table = features_table.lower()
    features_df = load_features_in_polygon(polygon, features_table, feature_names)
    image_mask_list = [get_image_overlay(features_df.iloc[:, -1], features_df[col], step=step, return_array=True) for
                       col in tqdm(features_df.columns[:-1], desc="Creating image from features", unit='feature')]
    images_list = [feature[0] for feature in image_mask_list]
    image = np.transpose(np.stack(images_list), (1, 2, 0))
    # create mask
    mask_index = image_mask_list[0][1]
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[mask_index] = 1
    return image, mask


def ors_is_up(host=ors_server_ip, port=ors_server_port) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


class FeatureFilter(TransformerMixin):
    # TODO: typing + add test
    # TODO: this is no longer used in pipeline, old bug
    def __init__(self, bundle: list=None, importance=10):
        if bundle is not None:
            from coord2vec.feature_extraction.features_builders import FeaturesBuilder
            from coord2vec.feature_extraction.feature_bundles import create_building_features
            all_feats = create_building_features(bundle, importance)
            builder = FeaturesBuilder(all_feats)
            self.feat_names = builder.all_feat_names


    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame):
        if self.feat_names is not None:
            assert isinstance(X, pd.DataFrame), f"X is not a DataFrame \n {X}"

            feat_names_after_filt = [c for c in X.columns if c in self.feat_names]
            if len(feat_names_after_filt) != len(self.feat_names):
                pass
                # logging.warning(f"""Some features in FeatureFilter do not appear in X
                #                     X: {X.columns}
                #                     filt_feat_names: {self.feat_names}
                #                 """)

            try:
                # TODO: old bug was that applying this after feature selection resulted in feature not being here
                X_filt = X[feat_names_after_filt]
            except:
                print("Error in FeatureFilter: returing X")
                X_filt = X
            return X_filt
        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return self.transform(X)
