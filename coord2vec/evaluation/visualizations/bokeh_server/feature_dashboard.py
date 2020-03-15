from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Row, LayoutDOM, Select, TextInput, Button
from shapely.geometry import Point

from coord2vec.common.geographic.geo_utils import get_closest_geo
from coord2vec.evaluation.visualizations.bokeh_plots import bokeh_multiple_score_maps_folium, \
    bokeh_score_histogram, log_current_map_position_js, bokeh_features_table
from coord2vec.feature_extraction.feature_utils import load_features_in_polygon
from coord2vec.feature_extraction.feature_table import GEOM_WKT


class FeatureDashboard:
    start_location = [36.483, 36.475]
    folium_zoom = 14

    def __init__(self, feature_names: List[str] = None):
        self.main_panel = column()
        self.folium_name = "features_folium"

        self.features_df = load_features_in_polygon(POLYGON, BUILDINGS_FEATURES_TABLE, feature_names).set_index(GEOM_WKT)
        # is_point = [isinstance(geo, Point) for geo in self.features_df.index]
        # import pdb
        # pdb.set_trace()
        self.features_df = self.features_df[[isinstance(geo, Point) for geo in self.features_df.index]]
        self.y_true = np.ones(len(self.features_df))
        self.main_panel.children.append(self.visualize_feature(self.features_df, self.y_true))

        # Bokeh elements
        self.feature_select = None

    def visualize_feature(self, features_df: pd.DataFrame, y_true: Iterable[int] = None) -> LayoutDOM:
        """
        Create a visualization for a feature table
        Args:
            features_df: the feature table, with features as columns
            y_true: the GT, optional

        Returns:
            A Bokeh layout of the feature visualization
        """
        # Add select
        self.feature_select = Select(title="Choose Feature: ",
                                     value=features_df.columns[0] if len(features_df) > 0 else 'Nada',
                                     options=list(features_df.columns), width=200)

        self.feature_select.js_on_change('value', log_current_map_position_js(self.folium_name))
        self.feature_select.on_change('value', self.choose_feature_callback)

        # add the feature values table
        lon_text = TextInput(value=str(self.start_location[1]), title='lon:', width=100)
        lat_text = TextInput(value=str(self.start_location[0]), title='lat:', width=100)
        self.lonlat_text_inputs = [lon_text, lat_text]
        self.sample_feature_button = Button(label='Get Sample Features')
        self.sample_feature_button.js_on_click(log_current_map_position_js(self.folium_name))
        self.sample_feature_button.on_click(self.update_bokeh_features_table)

        point = self.features_df.index[0]
        bokeh_features_table, _ = self.get_bokeh_features_table(point)

        # Add histogram and folium
        hist = bokeh_score_histogram(features_df.iloc[:, 0].dropna(), y_true[features_df.iloc[:, 0].notna().values])
        folium_fig = bokeh_multiple_score_maps_folium(features_df[[self.feature_select.value]], step=STEP_SIZE,
                                                      start_zoom=self.folium_zoom,
                                                      start_location=self.start_location, file_name=self.folium_name,
                                                      lonlat_text_inputs=self.lonlat_text_inputs)

        # put it all together
        self.folium_column = column(self.feature_select, row(hist, folium_fig))
        self.features_table = column(row(lon_text, lat_text), self.sample_feature_button, bokeh_features_table)

        return row(self.folium_column, self.features_table)

    def update_bokeh_features_table(self):
        lon = float(self.lonlat_text_inputs[0].value)
        lat = float(self.lonlat_text_inputs[1].value)
        point = Point([lon, lat])

        bokeh_features_table, closest_point = self.get_bokeh_features_table(point)

        self.lonlat_text_inputs[0].value = str(round(closest_point.x, 5))
        self.lonlat_text_inputs[1].value = str(round(closest_point.y, 5))
        self.features_table.children[-1] = bokeh_features_table

    def get_bokeh_features_table(self, point: Point):
        closest_point = get_closest_geo(point, self.features_df.index)
        closest_point_df = self.features_df[self.features_df.index == closest_point]
        bokeh_table = bokeh_features_table(closest_point_df.columns, closest_point_df.values)

        return bokeh_table, closest_point

    def choose_feature_callback(self, attr, old, new):
        print(f"Chose feature: {new}")
        new_hist = bokeh_score_histogram(self.features_df[new].dropna(),
                                         self.y_true[self.features_df[new].notna().values])
        new_folium = bokeh_multiple_score_maps_folium(self.features_df[[new]], step=STEP_SIZE,
                                                      start_zoom=self.folium_zoom,
                                                      start_location=self.start_location, file_name=self.folium_name,
                                                      lonlat_text_inputs=self.lonlat_text_inputs)

        self.folium_column.children[1] = row(new_hist, new_folium)
