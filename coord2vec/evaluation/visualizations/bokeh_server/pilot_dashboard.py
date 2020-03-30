import os
from typing import List

import numpy as np
import pandas as pd
import shap
from bokeh.layouts import column, row
from bokeh.models import Div
from shapely.geometry import Point
from shapely import wkt
from coord2vec.common.geographic.geo_utils import get_closest_geo
from coord2vec.evaluation.visualizations.bokeh_plots import bokeh_multiple_pr_curves, bokeh_score_histogram, \
    feature_importance, bokeh_scored_polygons_folium
from coord2vec.experiments.experiment_loader import load_separate_model_results


class PilotDashboard:
    start_location = None
    folium_zoom = 14

    def __init__(self, model_results_dir: str, pilot_results_dir: str, kfold_results_dir: str):
        # TODO change this when needed to load real results
        model_results_df = pd.read_csv(os.path.join(model_results_dir, '400_florentin.csv'))
        pilot_results_df = pd.read_csv(os.path.join(pilot_results_dir, 'pilot_florentin.csv'))
        self.combined_df = self._load_city(model_results_df, pilot_results_df)

        self.full_results = load_separate_model_results(kfold_results_dir)['model_results']
        self.model_results = self.full_results[self.model_idx]
        self.geos_train, self.X_train_df, self.y_train, self.train_probas, \
        self.geos_test, self.X_test_df, self.y_test, self.test_probas, \
        self.models, self.model_names, self.auc_scores = self._extract_model_results(self.full_results)

        self.folium_name = 'pilot_dashboard'
        self.main_panel = self.bokeh_plot()

    @staticmethod
    def _load_city(model_df, pilot_df):
        id_col = 'id'
        model_df = model_df.set_index(id_col)
        pilot_df = pilot_df.set_index(id_col)
        pilot_df = pilot_df.dropna(subset=['score'])

        combined_df = pilot_df.join(model_df, how='left')
        combined_df['geometry'] = combined_df['geometry'].apply(lambda x: wkt.loads(x))
        return combined_df

    def bokeh_plot(self):
        # create precision recall curve
        # test_probas_df = pd.DataFrame(
        #     {model_name: test_proba for model_name, test_proba in zip(self.model_names, self.test_probas)})
        pr_curve_df = self._create_pr_curve_data()
        pr_curve = bokeh_multiple_pr_curves(pr_curve_df, self.y_test.values)
        #
        # # create feature histogram
        # self.feature_select = Select(title="Choose Feature: ", value=self.X_train_df.columns[0],
        #                              options=list(self.X_train_df.columns), width=200)
        # self.feature_select.on_change('value', self.choose_feature_callback)
        # # self.feature_select.js_on_change('value', log_current_map_position_js(self.folium_name))
        # hist = bokeh_score_histogram(self.X_train_df.iloc[:, 0], self.y_train > 0.5)
        #
        # # create model select
        # self.model_select = Select(title="Choose Model: ", value=self.model_names[0],
        #                            options=list(self.model_names), width=300)
        # self.kfold_select = Select(title="Choose Kfold: ", value='0',
        #                            options=list(map(str, range(self.num_kfold))), width=150)
        # self.run_button = Button(label="RUN", button_type="success", width=100)
        # self.run_button.js_on_click(log_current_map_position_js(self.folium_name))
        # self.run_button.on_click(self.run_callback)
        #
        # # build input for map click coords
        # lon_text = TextInput(value="0", title='lon:', width=100)
        # lat_text = TextInput(value="0", title='lat:', width=100)
        # self.lonlat_text_inputs = [lon_text, lat_text]
        # feature_importance_button = Button(label='Calculate Feature Values')
        # # feature_importance_button.js_on_click(log_current_map_position_js(self.folium_name))
        # feature_importance_button.on_click(self.sample_feature_importance_update)
        # importance_fig, _ = self.build_importance_figure(Point(0, 0))

        # create folium figure
        scores_columns = ['inteligence_score', 'ground_score', 'gt']
        folium_series = self._create_folium_series(self.combined_df, scores_columns)
        # TODO add the cities once back to LOTR
        folium_fig = bokeh_scored_polygons_folium(folium_series,
                                                  [True] * len(folium_series), start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700, image_resolution=10)

        # put it all together
        # self.left_column = column(pr_curve, self.feature_select, hist)
        # self.folium_column = column(row(self.model_select, self.kfold_select, self.run_button), folium_fig,
        #                             self.mean_auc)
        # self.importance_and_val_column = column(row(lon_text, lat_text), feature_importance_button, importance_fig)
        return row(pr_curve, folium_fig)

        # return row(self.left_column, self.folium_column, self.importance_and_val_column)

    def sample_feature_importance_update(self):
        lat, lon = self.get_click_lonlat()
        point = Point([lon, lat])
        importance_figure, closest_geo = self.build_importance_figure(point)

        self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        self.importance_and_val_column.children[-1] = importance_figure

    def choose_feature_callback(self, attr, old, new):
        print(f"Chose feature: {new}")
        new_hist = bokeh_score_histogram(self.X_train_df[new], self.y_train > 0.5)
        self.left_column.children[-1] = new_hist

    def build_importance_figure(self, point: Point):
        if point.x == 0 and point.y == 0:
            closest_geo = point
            closest_geo_df = pd.DataFrame([self.features_df.mean().values], columns=self.features_df.columns,
                                          index=['aggregated'])
        else:
            closest_geo = get_closest_geo(point, self.features_df.index)
            closest_geo_df = self.features_df[self.features_df.index == closest_geo]

        model = self.models[self.model_names.index(self.model_select.value)]
        explainer = shap.KernelExplainer(model.predict, self.features_kmeans)
        shap_values_model = explainer.shap_values(closest_geo_df, silent=True)[0]  # TODO: very slow
        # importance_fig = shap.force_plot(explainer.expected_value, shap_values_model, closest_geo_df)

        sort_indices = np.argsort(np.abs(shap_values_model))
        importance_fig = feature_importance(list(self.X_train_df.columns[sort_indices]),
                                            shap_values_model[sort_indices])

        # shap.save_html('importance_fig.html', importance_fig)
        # importance_fig = Div(text="""<iframe>
        #                                 src="importance_fig.html"
        #                             </iframe>""")

        return importance_fig, closest_geo

    def run_callback(self):
        self.model_idx = self.model_names.index(self.model_select.value)
        self.kfold = int(self.kfold_select.value)

        # set all the parameters of the new kfold
        self.geos_train, self.X_train_df, self.y_train, self.train_probas, \
        self.geos_test, self.X_test_df, self.y_test, self.test_probas, \
        self.models, self.model_names, self.auc_scores = self._extract_model_results(self.full_results, self.model_idx,
                                                                                     self.kfold)

        self.num_kfold = len(self.full_results[0]['train_idx'])
        self.features_df = pd.concat([self.X_train_df, self.X_test_df])
        self.features_kmeans = shap.kmeans(self.features_df, 10)
        self.all_probas = np.concatenate([self.train_probas[self.model_idx], self.test_probas[self.model_idx]])
        self.all_probas_df = pd.Series(data=self.all_probas, index=self.features_df.index)

        # create new precision recall curve
        test_probas_df = pd.DataFrame(
            {model_name: test_proba for model_name, test_proba in zip(self.model_names, self.test_probas)})
        pr_curve = bokeh_multiple_pr_curves(test_probas_df, self.y_test.values)
        self.left_column.children[0] = pr_curve

        # create new folium map
        print(f"Choose model: {self.model_select.value}\t kfold: {self.kfold}")
        new_folium = bokeh_scored_polygons_folium([self.all_probas_df],
                                                  [True], train_geos=self.geos_train,
                                                  test_geos=self.geos_test, start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700, lonlat_text_inputs=self.lonlat_text_inputs)
        self.folium_column.children[-2] = new_folium

        self.sample_feature_importance_update()

    def get_click_lonlat(self):
        lon = self.lonlat_text_inputs[0].value
        lat = self.lonlat_text_inputs[1].value
        lon = float(lon) if lon != "" else 0
        lat = float(lat) if lat != "" else 0
        return lat, lon

    def _extract_model_results(self, full_results):
        train_kfold, test_kfold = 0, 2
        model_results = full_results[0]
        train_idx, test_idx = model_results['train_idx'][train_kfold], model_results['test_idx'][test_kfold]

        # X and y
        X_train_df, X_test_df = model_results['X_df'].iloc[train_idx], model_results['X_df'].iloc[test_idx]
        y_train, y_test = model_results['y'][train_idx], model_results['y'][test_idx]

        # geos
        geos_train_idx, geos_test_idx = model_results['geos_kfold_split'][test_kfold]
        geos_train, geos_test = model_results['geos'][geos_train_idx], model_results['geos'][geos_test_idx]

        # probas
        train_probas = [results['probas'][train_kfold][train_idx] for results in full_results]
        test_probas = [results['probas'][test_kfold][test_idx] for results in full_results]

        # models
        model_names = [results['model_name'] for results in full_results]
        models = [model_results['models'][train_kfold], model_results['models'][train_kfold]]
        scores = [results['auc_scores'] for results in full_results]

        return geos_train, X_train_df, y_train, train_probas, \
               geos_test, X_test_df, y_test, test_probas, \
               models, model_names, scores

    def _create_pr_curve_data(self):
        probas_df = self.combined_df[['geometry', 'score']].set_index('geometry').rename(columns={'score': 'human'})
        probas_df['train'] = probas_df.join(self.train_probas, how='inner').iloc[:, -1]
        probas_df['test'] = probas_df.join(self.test_probas, how='inner').iloc[:, -1]
        return probas_df

    def _create_folium_series(self, combined_df: pd.DataFrame, scores_columns: List[str]):
        new_df = combined_df.reset_index().set_index('geometry')
        folium_series = [new_df['score'] - new_df[score_col] for score_col in scores_columns]
        folium_series = [pd.Series(data=series, index=new_df.index, name=score_col) for series, score_col in
                         zip(folium_series, scores_columns)]
        folium_series = [series.dropna() for series in folium_series]
        folium_series = [series - series.min() / (series.max() - series.min()) for series in folium_series]
        return folium_series
