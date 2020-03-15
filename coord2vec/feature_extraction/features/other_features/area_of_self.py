from coord2vec.common.geographic.geo_utils import area_in_m2
from coord2vec.feature_extraction.feature import Feature
import geopandas as gpd
import pandas as pd


class AreaOfSelf(Feature):
    def __init__(self, **kwargs):
        table_filter_dict = {}
        feature_name = f"area_of_self"
        self.radius = 0
        super().__init__(table_filter_dict=table_filter_dict, feature_names=[feature_name], radius=0,**kwargs)

    def _build_postgres_query(self):
        return

    def _calculate_feature(self, input_gs: gpd.GeoSeries):
        # TODO: do we want to push to postgresql and calculate it there ? might be faster (epsecially if we already pushed the geoms for other features

        # edit the df
        areas = [area_in_m2(geo) for geo in input_gs]
        full_df = pd.DataFrame(data={'geom': input_gs, self.feature_names[0]: areas}, index=range(len(input_gs)))

        return full_df

    def set_default_value(self, radius):
        self.default_value = {self.feature_names[0]: 0}
