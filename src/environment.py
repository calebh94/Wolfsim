"""
Environment definition modified from: https://github.com/jrmak/FNNR-ABM-Primate/
# Land categories are typed as classes to keep up with in mesa grid object
"""

from mesa.agent import Agent

class Environment(Agent):

    def __init__(self, unique_id, model, pos = None, elevation = None):
        super().__init__(unique_id, model)
        self.pos = pos
        self.elevation = elevation

    def step(self):
        pass

class Vegetation(Environment):
    type = 0

class Outside_FNNR(Environment):
    type = -9999

# elevation
class Elevation_Out_of_Bound(Environment):
    type = 99
    lower_bound = 1
    upper_bound = 3000

class Img(Environment):
    type = 1