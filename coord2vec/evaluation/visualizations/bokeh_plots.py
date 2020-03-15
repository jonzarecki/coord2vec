import itertools
import logging
import os
import time
from typing import List, Tuple, Dict, Iterable

import bokeh
import folium
import numpy as np
import pandas as pd
from bokeh.layouts import LayoutDOM
from bokeh.models import HoverTool, TableColumn, DataTable
from bokeh.models import HoverTool, ColumnDataSource, Div, Row, Column, CustomJS
from bokeh.models.widgets import Select
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from scipy.stats.kde import gaussian_kde
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from sklearn.metrics import precision_recall_curve
import geopandas as gpd
from coord2vec.common.geographic.geo_map import GeoMap
from coord2vec.common.geographic.geo_utils import meters2degrees
from coord2vec.common.geographic.visualization_utils import get_image_overlay, LatLngCopy
import seaborn as sns

from coord2vec.evaluation.evaluation_metrics.metrics import soft_precision_recall_curve


def bokeh_pr_curve_raw(precision: Iterable[float], recall: Iterable[float], threshold: List[float], fig: figure = None,
                       color: str = 'blue', legend=None) -> LayoutDOM:
    """
    plot  precision recall curve. can add to an existing figure, or create a new one
    Args:
        precision: iterable of precision points
        recall: iterable of recall points
        threshold: iterable of threshold points which resulted in the previous $precision and $recall
        fig: if None: create a new bokeh figure, otherwise, add plot to figure one
        color: color of the plot

    Returns: precision recall bokeh plot
    """
    if fig is None:
        hover = HoverTool(
            tooltips=[('recall', "$x{%0.2f}"), ('precision', "$y{%0.2f}"), ('threshold', "@threshold{%0.2f}")])
        tools = "pan, wheel_zoom, box_zoom, reset"
        fig = figure(x_range=(0, 1), y_range=(0, 1), tools=[hover, tools], title='Precision Recall Curve',
                     plot_height=350, plot_width=400)

    cds = ColumnDataSource(data={'precision': precision, 'recall': recall, 'threshold': threshold})
    fig.line(x='recall', y='precision', line_width=2, source=cds, color=color, muted_alpha=0.2,
             muted_color=color)  # legend=legend
    # fig.circle(x='precision', y='recall', line_width=1, source=cds)
    fig.xaxis.axis_label = 'Recall'
    fig.yaxis.axis_label = 'Precision'
    fig.legend.click_policy = "mute"
    fig.legend.label_text_font_size = '6pt'
    return fig


def bokeh_pr_curve_from_y_proba(y_proba: Iterable[float], y_true: Iterable[bool], fig: figure = None,
                                color: str = 'blue', legend="Default") -> LayoutDOM:
    """
    plot  precision recall curve. can add to an existing figure, or create a new one
    Args:
        y_proba: iterable of predicted labels
        y_true: iterable of true labels
        fig: if None: create a new bokeh figure, otherwise, add plot to figure one
        color: color of the plot

    Returns: precision recall bokeh plot
    """
    precision, recall, thresholds = soft_precision_recall_curve(y_true, y_proba)
    return bokeh_pr_curve_raw(precision, recall, thresholds, fig, color, legend)


def bokeh_multiple_pr_curves(y_probas_df: pd.DataFrame, y_true: Iterable[bool]) -> LayoutDOM:
    """
    Create multiple precision-recall curves for multiple models_dict
    Args:
        y_probas_df: scores of different models_dict, with columns as models_dict
        y_true: the GT

    Returns:
        A Bokeh plot with multiple pr curves
    """
    fig = None
    colors = itertools.cycle(palette)
    for column in y_probas_df.columns:
        fig = bokeh_pr_curve_from_y_proba(y_probas_df[column], y_true, fig=fig, color=colors.__next__(), legend=column)
    return fig


def bokeh_multiple_score_maps_folium(scores_df: pd.DataFrame,
                                     train_geos: Iterable[BaseGeometry] = None,
                                     test_geos: Iterable[BaseGeometry] = None,
                                     start_zoom: int = 9,
                                     height: int = 600,
                                     width: int = 850,
                                     step: float = 50,
                                     start_location=(36.4, 36.4),
                                     file_name: str = 'folium',
                                     lonlat_text_inputs: list = None) -> Tuple[LayoutDOM, GeoMap]:
    """
    create a folium map of different scores per coordinate

    Args:
        scores_df: dataframe with index of type Point, and columns of different scores or features
        train_geos: Train geometries to plot
        test_geos: Test geometries to plot
        start_zoom: start zoom of folium
        height: height of the folium map in pixels
        width: width of the folium map in pixels
        step: step size of the interpolation in meters
        start_location: start location of the geomap
        file_name: file name for the html file saving the folium map. Should be different for every map
        lonlat_text_inputs: text_input for lon and lat synchronizing
    Returns:
        A map with all the different scores on it.
    """
    geo_map = GeoMap(start_zoom=start_zoom, start_location=start_location)

    # Add the score layers
    for i, column in enumerate(scores_df.columns):
        layer = get_image_overlay(scores_df.index, scores_df[column], step=step, name=column)
        if i > 0:
            layer.show = False
        layer.add_to(geo_map.map)

    # add the polygons
    if train_geos is not None:
        train_geos_df = pd.DataFrame({"geos": [geo.wkt for geo in train_geos]})
        geo_map.load_wkt_layer_from_dataframe(train_geos_df, wkt_column_name="geos", fill_alpha=0, color='blue',
                                              group_name='train')
    if test_geos is not None:
        test_geos_df = pd.DataFrame({"geos": [geo.wkt for geo in test_geos]})
        geo_map.load_wkt_layer_from_dataframe(test_geos_df, wkt_column_name="geos", fill_alpha=0, color='purple',
                                              group_name='test')

    # set some parameters for the map
    folium.LayerControl().add_to(geo_map.map)
    folium.plugins.MeasureControl().add_to(geo_map.map)
    folium.plugins.MousePosition(lng_first=True).add_to(geo_map.map)
    LatLngCopy().add_to(geo_map.map)

    # save the folium html
    static_folder = os.path.join(os.path.dirname(__file__), 'bokeh_server', 'static')
    os.makedirs(static_folder, exist_ok=True)
    geo_map.map.save(os.path.join(static_folder, f'{hash(file_name)}.html'))
    click_str = f"""
                f.contentWindow.document.body.onclick = 
                function() {{
                    ff = document.getElementById('{hash(file_name)}');
                    map = eval('ff.contentWindow.'+ff.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                    window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[0].id}].value = map.loc[1].toFixed(5).toString();
                    window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[1].id}].value = map.loc[0].toFixed(5).toString();
                }};
                """ if lonlat_text_inputs is not None else ""
    fig = Div(text=f"""
    <iframe onload="console.log('changing map props');
                f = document.getElementById('{hash(file_name)}');
                map = eval('f.contentWindow.'+f.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                    if(self.center && self.zoom){{map.setView(self.center, self.zoom);}};
                                    
                {click_str}
                "
                
            id="{hash(file_name)}"
            src="bokeh_server/static/{hash(file_name)}.html?brus=stp{time.time()}"
            width={width} height={height}></iframe>
    """, height=height, width=width)

    return fig


color_gradients = [
    # green to red
    LinearSegmentedColormap.from_list('rgb', [(0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0)], N=256, gamma=1.0),
    # deep blue to light blue
    LinearSegmentedColormap.from_list('rgb', [(0.20442906574394465, 0.29301038062283735, 0.35649365628604385, 0),
                                              (0.20898116109188775, 0.38860438292964244, 0.5173343585800334, 0),
                                              (0.21341022683583238, 0.4816147635524798, 0.6738280148660771, 0),
                                              (0.2818813276944765, 0.5707599641163655, 0.7754914776368064, 0),
                                              (0.41069332308086637, 0.6512213251313598, 0.8168294245802896, 0),
                                              (0.5430834294502115, 0.733917723952326, 0.8593156478277586, 0)],
                                      N=256, gamma=1.0)]


def bokeh_scored_polygons_folium(poly_scores_list: List[pd.Series],
                                 use_image_list: List[bool],
                                 train_geos: gpd.GeoSeries = None,
                                 test_geos: gpd.GeoSeries = None,
                                 start_zoom: int = 9,
                                 height: int = 600,
                                 width: int = 850,
                                 start_location=(36.4, 36.4),
                                 file_name: str = 'folium',
                                 lonlat_text_inputs: list = None,
                                 image_resolution = 2,
                                 return_geomap:bool = False):
    """
        create a folium map of different scores per coordinate

        Args:
            poly_scores_list: List of dataframes with index of type Polygon, and columns of
                                different scores or features
            use_image_list: List of boolean matching to poly_scores_list, True if we want to use image overlay to display
            train_geos: Train geometries to plot
            train_geos: Test geometries to plot
            start_zoom: start zoom of folium
            height: height of the folium map in pixels
            width: width of the folium map in pixels
            start_location: start location of the geomap
            file_name: file name for the html file saving the folium map. Should be different for every map
            lonlat_text_inputs: text_input for lon and lat synchronizing
        Returns:
            A map with all the different scores on it.
        """
    geo_map = GeoMap(start_zoom=start_zoom, start_location=start_location)

    assert len(poly_scores_list) == len(use_image_list), "both lists should be the same length"
    if len(poly_scores_list) > 2:
        logging.warning("no colors gradients after 2 lists")

    for poly_score_df, use_image, cm in zip(poly_scores_list, use_image_list, color_gradients):
        poly_df = pd.DataFrame({'geos': [poly.buffer(meters2degrees(50)).wkt if isinstance(poly, Point) else poly.wkt for poly in poly_score_df.index]})
        colors = [rgb2hex(color) for color in cm(poly_score_df)[:, :3]]
        if use_image:
            geo_map.load_image_overlay_from_dataframe(poly_df, wkt_column_name="geos", fill_alpha=0.2, color=colors,
                                                      name='scores', fill_color=colors, step=image_resolution, line_alpha=1.00)
        else:
            geo_map.load_wkt_layer_from_dataframe(poly_df, wkt_column_name="geos", fill_alpha=0.2, color=colors,
                                                  group_name='scores', fill_color=colors)

    # add the polygons
    if train_geos is not None:
        train_geos_df = pd.DataFrame({"geos": [geo.wkt for geo in train_geos]})
        geo_map.load_wkt_layer_from_dataframe(train_geos_df, wkt_column_name="geos", fill_alpha=0, color='black',
                                              group_name='train', pop_up=False)

    if test_geos is not None:
        test_geos_df = pd.DataFrame({"geos": [geo.wkt for geo in test_geos]})
        geo_map.load_wkt_layer_from_dataframe(test_geos_df, wkt_column_name="geos", fill_alpha=0, color='purple',
                                              group_name='test', pop_up=False)

    # set some parameters for the map
    folium.LayerControl().add_to(geo_map.map)
    folium.plugins.MeasureControl().add_to(geo_map.map)
    folium.plugins.MousePosition(lng_first=True).add_to(geo_map.map)
    LatLngCopy().add_to(geo_map.map)

    # save the folium html
    static_folder = os.path.join(os.path.dirname(__file__), 'bokeh_server', 'static')
    os.makedirs(static_folder, exist_ok=True)
    geo_map.map.save(os.path.join(static_folder, f'{hash(file_name)}.html'))

    click_str = f"""
                f.contentWindow.document.body.onclick = 
                function() {{
                    ff = document.getElementById('{hash(file_name)}');
                    map = eval('ff.contentWindow.'+ff.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                    window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[0].id}].value = map.loc[1].toFixed(5).toString();
                    window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[1].id}].value = map.loc[0].toFixed(5).toString();
                }};
                """ if lonlat_text_inputs is not None else ""
    fig = Div(text=f"""
        <iframe onload="console.log('changing map props');
                    f = document.getElementById('{hash(file_name)}');
                    map = eval('f.contentWindow.'+f.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                    if(self.center && self.zoom){{map.setView(self.center, self.zoom);}};
                    
                    {click_str}
                    "
                id="{hash(file_name)}"
                src="bokeh_server/static/{hash(file_name)}.html?brus=stp{time.time()}"
                width={width} height={height}></iframe>
        """, height=height, width=width)
    if return_geomap:
        return geo_map
    return fig


def bokeh_score_histogram(y_proba: Iterable[float], y_true: Iterable[float]):
    """
    plot score histogram curve. can add to an existing figure, or create a new one
    Args:
        y_proba: iterable of predicted labels
        y_true: iterable of true labels

    Returns: score histogram bokeh plot
    """
    x_range = (min(y_proba), max(y_proba))

    hover = HoverTool(tooltips=[('value', "$x{0.2f}"), ('probability', "$y{0.2f}")])
    tools = "pan, wheel_zoom, box_zoom, reset"
    #TODO: .name ? fix this and typing
    fig = figure(x_range=x_range, y_range=(0, 1), tools=[hover, tools], title=f"{y_proba.name} Histogram",
                 plot_height=400, plot_width=400)

    max_hist = 0
    for label in np.unique(y_true):
        color = 'blue' if label == 0 else 'red'
        cur_y_proba = y_proba[y_true == label]
        hist, edges = np.histogram(cur_y_proba, density=True, bins=20, range=x_range)
        fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4, color=color, legend=str(label))

        x = np.linspace(*x_range, 200)
        try:
            fig.line(x, gaussian_kde(cur_y_proba)(x), color=color, legend=str(label))
        except:
            pass
        max_hist = max(max_hist, max(hist))

    fig.y_range = bokeh.models.Range1d(0, max_hist * 1.2)
    fig.legend.click_policy = "mute"
    return fig


def feature_importance(feature_names: List[str], feature_importance: np.ndarray) -> LayoutDOM:
    """
    Args:
        feature_names: names of all the features
        feature_importance: the importance of the features

    Returns:
        A bokeh hbar of the feature importance
    """
    source = ColumnDataSource(data={'feature_names': feature_names, 'feature_importance': feature_importance})

    hover = HoverTool(tooltips=[('feature', "@feature_names"),
                                ('weight', "@feature_importance{0.00}")])
    fig = figure(title='Feature Importance', y_range=feature_names, toolbar_location=None, plot_height=800,
                 plot_width=300, tools=[hover])
    fig.hbar(y='feature_names', right='feature_importance', height=0.95, source=source)
    fig.xaxis.axis_label = 'importance'
    # fig.yaxis.major_label_orientation = 45
    fig.yaxis.major_label_text_font_size = '0pt'

    # remove y tick
    return fig


def log_current_map_position_js(file_name: str = "folium"):
    return CustomJS(args=dict(), code=f"""
                    f = document.getElementById('{hash(file_name)}')
                    map = eval('f.contentWindow.'+f.contentWindow.document.getElementsByClassName('folium-map')[0].id)

                    self.center = [map.getCenter()['lat'], map.getCenter()['lng']];
                    self.zoom = map.getZoom()
                    """)


def bokeh_features_table(feature_names: List[str], feature_values: Iterable[float]) -> LayoutDOM:
    """
    Create a 2-columns table that shows features' values
    Args:
        feature_names: The names of the features
        feature_values: The values of the features

    Returns:
        A Bokeh layout object of a table
    """
    source = ColumnDataSource({'feature_values': feature_values, 'feature_names': feature_names})
    columns = [TableColumn(field='feature_names', title='Feature'),
               TableColumn(field='feature_values', title='Value', width=50)]
    data_table = DataTable(source=source, columns=columns, header_row=True, width=250, height=600)
    return data_table
