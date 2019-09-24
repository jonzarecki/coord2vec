import unittest

import openrouteservice


class TestOrsServer(unittest.TestCase):
    def test_server_running(self):
        coords = ((34.482724, 31.492354), (34.492724, 31.452354))
        print("")

        # key can be omitted for local host
        client = openrouteservice.Client(base_url='http://52.236.160.138:8080/ors')

        # Only works if you didn't change the ORS endpoints manually
        routes = client.directions(coords, instructions=False, geometry=False)
        self.assertIsNotNone(routes)


if __name__ == '__main__':
    unittest.main()
