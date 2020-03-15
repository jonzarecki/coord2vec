from abc import ABC

import geopandas as gpd
import pandas as pd

from coord2vec.common.db.postgres import get_df, connect_to_db
from coord2vec.feature_extraction.feature import Feature


class BasePostgresFeature(Feature, ABC):
    def __init__(self, **kwargs):
        """

        Args:
            table_filter_dict: a dictionary of shape: {table_name: {filter_name: filter_sql}}
                                should contain all the tables, with all the filters required for this filter.
        """
        super().__init__(**kwargs)

    def _calculate_feature(self, input_gs: gpd.GeoSeries):
        if self.intersect_tbl_name_dict is None or self.input_geom_table is None:
            raise Exception("Must use an OSM feature factory before extracting the feature")

        # calculate the feature
        conn = connect_to_db()
        query = self._build_postgres_query()
        res = get_df(query, conn=conn)

        # edit the df
        full_df = pd.DataFrame(index=range(len(input_gs)), columns=self.feature_names)
        if len(res['geom_id']) != 0:
            full_df.iloc[res['geom_id']] = res.drop('geom_id', axis=1).values
        full_df.fillna(self.default_value, inplace=True)
        full_df['geom'] = input_gs

        conn.close()

        return full_df
