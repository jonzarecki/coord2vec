import openrouteservice

from coord2vec.feature_extraction.feature import Feature


class OrsFeature(Feature):
    """
    This class filters objects from the ors service.
    It gives nice methods for calling ors functions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = openrouteservice.Client(base_url='http://52.236.160.138:8080/ors')

