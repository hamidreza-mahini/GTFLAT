# Testing main functions using the Hawk Dove and Hawk Dove Bourgeois games

from wrapper import GameDynamicsWrapper, VariedGame
from dynamics.wright_fisher import WrightFisher
from games.example_games.hawk_dove import HawkDove
from games.example_games.hdb import HawkDoveBourgeois


import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        import logging
        logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    def test_single_simulation(self):
        s = GameDynamicsWrapper(HawkDove, WrightFisher)
        s.simulate(num_gens=100, graph=dict(options=['area', 'smallFont']))

    def test_many_simulation(self):  # Determines which equilibria result based upon several simulations, text output
        s = GameDynamicsWrapper(HawkDove, WrightFisher)
        print(s.simulate_many(num_iterations=100, num_gens=100, graph=dict(options=['area', 'smallFont'])))

    def test_vary_one(self):  # Simulates while changing a single variable over time
        s = VariedGame(HawkDoveBourgeois, WrightFisher, dynamics_kwargs=dict(uniDist=True))
        s.vary_param('v', (0, 5, 1), num_gens=500, num_iterations=5, graph=dict(options=['area']))

    def test_wireFrame(self):  # 3d graph of equilibrium found when varying two variables
        s = VariedGame(HawkDoveBourgeois, WrightFisher)
        s.vary_2params('v', (0, 5, 1), 'c', (1, 5, 1), num_iterations=1, num_gens=200)

    def test_contour_graph(self):  # 2d contour color plot
        s = VariedGame(HawkDoveBourgeois, WrightFisher)
        s.vary_2params('v', (0, 50, 1), 'c', (0, 100, 1), num_iterations=1, num_gens=500, burn=499, graph=dict(type='contour', lineArray=[(0, 50, 0, 50)]))

if __name__ == '__main__':
    unittest.main()
