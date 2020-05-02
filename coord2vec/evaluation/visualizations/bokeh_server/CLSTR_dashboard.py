import logging

from bokeh.layouts import column, row
from bokeh.models import Select, Button, Slider, HoverTool
import numpy as np

from coord2vec.config import CLSTR_RESULTS_DIR
# from coord2vec.evaluation.evaluation_metrics.detection import calc_roc_for_th, calc_mAP
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.evaluation.visualizations.bokeh_plots import log_current_map_position_js, bokeh_scored_polygons_folium, \
    bokeh_pr_curve_raw
from coord2vec.experiments.experiment_loader import load_experiment_results


class CLSTRDashboard:
    start_location = [36.483, 36.475]
    folium_zoom = 14

    def __init__(self, task: TaskHandler, results_dir=CLSTR_RESULTS_DIR):
        self.task = task
        self.main_panel = column()
        self.folium_name = str(type(task))

        self.fold_params = ["building_scores", "train_geos", "y_train_buildings", "test_geos", "test_true_geos",
                            "y_test_buildings", "best_potential_CLSTRs_test"]
        # unpack CLSTR experiment results
        self.kfold_results = load_experiment_results(results_dir)['model_results']
        self.kfold = 0
        self.show_top_percent = 100
        curr_kfold_results = self.kfold_results[self.kfold]
        self.best_test_buildings_with_scores, self.train_geos, \
        self.y_train_buildings, self.test_geos, self.test_true_geos, self.y_test_buildings, self.best_potential_CLSTRs_test = \
            [curr_kfold_results[key] for key in self.fold_params]

        # TODO: assuming that the last test-geo is the false one (need to pass properly from pipe)
        self.true_CLSTR_geos = self.test_geos[:-1]

        plot = self.bokeh_plot()
        self.main_panel.children.append(plot)

    def bokeh_plot(self):
        # create folium figure
        self.kfold_select = Select(title="Choose Kfold: ", value='0',
                                   options=list(map(str, range(len(self.kfold_results)))), width=150)
        self.top_percent_CLSTRs_slider = Slider(start=0, end=100, step=5, value=100, title="Show top CLSTRs %")
        self.run_button = Button(label="RUN", button_type="success", width=100)
        self.run_button.js_on_click(log_current_map_position_js(self.folium_name))
        self.run_button.on_click(self.run_callback)

        selected_CLSTRs = self.best_potential_CLSTRs_test.iloc[
                       :int((self.show_top_percent / 100.) * len(self.best_potential_CLSTRs_test))]

        selected_CLSTRs_geos = selected_CLSTRs.index.tolist()
        for t_geo in self.true_CLSTR_geos:
            pass
            # calculate for each t_geo its intersection with the CLSTRs

        folium_fig = bokeh_scored_polygons_folium([self.best_test_buildings_with_scores, selected_CLSTRs], [True, False],
                                                  train_geos=self.train_geos, test_geos=self.test_geos,
                                                  start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700)

        self.folium_row = row(folium_fig, self.build_pr_curve())#, self.bokeh_percentile_to_mAP())
        self.kfold_select_row = row(self.kfold_select, self.top_percent_CLSTRs_slider, self.run_button)

        return column(self.kfold_select_row, self.folium_row)

    def build_pr_curve(self):
        logging.info("calculating pr curve to current CLSTRs")
        selected_CLSTRs = self.best_potential_CLSTRs_test.iloc[
                       :int((self.show_top_percent / 100.) * len(self.best_potential_CLSTRs_test))]

        prec, rec, ths = calc_roc_for_th(np.linspace(0, 1, 10, endpoint=True), selected_CLSTRs.index,
                                         selected_CLSTRs.values.tolist(), self.test_geos)
        return bokeh_pr_curve_raw(prec, rec, ths)

    def bokeh_percentile_to_mAP(self):  # semi working
        logging.info("calculating percentile to mAP")
        mean_ap_l = []
        for n in range(0, 101, 1):
            selected_CLSTRs = self.best_potential_CLSTRs_test.iloc[:int((n / 100.) * len(self.best_potential_CLSTRs_test))]
            mean_ap = calc_mAP(selected_CLSTRs.index, selected_CLSTRs.values.tolist(), self.test_geos)
            mean_ap_l.append(mean_ap)
        tools = "pan, wheel_zoom, box_zoom, reset"
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource
        hover = HoverTool(tooltips=[('percentile', "$x{%0.2f}"), ('mAP', "$y{%0.2f}")])
        fig = figure(x_range=(0, 1), y_range=(0, 1), tools=[hover, tools], title='Percentile to mAP',
                     plot_height=350, plot_width=400)

        cds = ColumnDataSource(data={'mean_ap': mean_ap_l, 'percentile': list(range(0, 101, 1))})
        fig.line(x='percentile', y='mean_ap', line_width=2, source=cds, color='blue', muted_alpha=0.2,
                 muted_color='blue')
        fig.xaxis.axis_label = 'percentile'
        fig.yaxis.axis_label = 'mAP'
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = '6pt'
        return fig

    def run_callback(self):
        # set all the parameters of the new kfold
        self.kfold = int(self.kfold_select.value)
        self.show_top_percent = int(self.top_percent_CLSTRs_slider.value)
        curr_kfold_results = self.kfold_results[self.kfold]
        self.best_test_buildings_with_scores, self.train_geos, \
        self.y_train_buildings, self.test_geos, self.test_true_geos, self.y_test_buildings, self.best_potential_CLSTRs_test = \
            [curr_kfold_results[key] for key in self.fold_params]

        selected_CLSTRs = self.best_potential_CLSTRs_test.iloc[
                       :int((self.show_top_percent / 100.) * len(self.best_potential_CLSTRs_test))]
        # create new folium map
        print(f"Choose kfold: {self.kfold}")
        new_folium = bokeh_scored_polygons_folium([self.best_test_buildings_with_scores, selected_CLSTRs], [True, False],
                                                  train_geos=self.train_geos, test_geos=self.test_geos,
                                                  start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700)
        self.folium_row.children[0] = new_folium
        self.folium_row.children[1] = self.build_pr_curve()
