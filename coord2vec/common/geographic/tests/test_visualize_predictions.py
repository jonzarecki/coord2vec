import unittest

from coord2vec.common.geographic.visualizations import visualize_predictions
from coord2vec import config
from flask import Flask
import numpy as np
import folium

app = Flask(__name__)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        l = (34.482724, 31.492354)
        map = visualize_predictions(np.array([(l[0]+i*0.005, l[1]+i*0.005) for i in range(500)]),
                              np.random.random((500,)) - 0.5)

        @app.route('/')
        def index():
            # start_coords = (46.9540700, 142.7360300)
            # folium_map = folium.Map(location=start_coords, zoom_start=14)
            return map._repr_html_()

        app.run(debug=True)

if __name__ == '__main__':
    unittest.main()
