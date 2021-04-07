"""
Wolfsim Model using the Mesa Agent-based Python Framework

Wolfsim Model defines the agent-based simulation components such as the creation of the agents, form of the grid, etc.

"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from numpy import random
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np

from WolfsimAgents import WolfAgent
from environment import *


def compute_pack_health(model):
    agent_ages = [agent.age for agent in model.schedule.agents]
    agent_health = [agent.alive for agent in model.schedule.agents]

    agesum = np.sum(agent_ages[agent_health==True]) / len(agent_health) #TODO: not calculccated right?
    return agesum

# DATA FILES
vegetation_file = 'data/vegetation.txt'  # change these filenames to another file in the same directory as needed
elevation_file = 'data/DEM.txt'
hh_file = 'data/hh_survey.csv'
resource_dict = {}


def _readCSV(text):
    # reads in a .csv file.
    # separate from _readASCII in model.py, which reads .asc files.
    cells = []
    f = open(text, 'r')
    body = f.readlines()
    for line in body:
        cells.append(line.split(","))
    return cells


class Resource(Agent):
    # Resources are fuelwood, mushrooms, herbs, etc. (see 'type') that humans collect.
    # They are considered agents in case a future version of the model makes them limited (accounts for land change).
    def __init__(self, unique_id, model, position, hh_id_match, type, frequency):
        super().__init__(unique_id, model)
        self.position = position
        self.hh_id_match = hh_id_match
        self.type = type
        self.frequency = frequency

    def step(self):
        pass



class WolfModel(Model):
    """A model with some number of wolves."""
    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        width = self._readASCII(vegetation_file)[1] # width as listed at the beginning of the ASCII file
        height = self._readASCII(vegetation_file)[2] # height as listed at the beginning of the ASCII file
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.time = 0

        empty_masterdict = {'Outside_FNNR': [], 'Elevation_Out_of_Bound': [], 'Household': [], 'PES': [], 'Farm': [],
                            'Forest': [], 'Bamboo': [], 'Coniferous': [], 'Broadleaf': [], 'Mixed': [], 'Lichen': [],
                            'Deciduous': [], 'Shrublands': [], 'Clouds': [], 'Farmland': []}

        # TODO: ONLY DO FIURST RUN
        gridlist = self._readASCII(vegetation_file)[0]  # list of all coordinate values; see readASCII function
        gridlist2 = self._readASCII(elevation_file)[0]  # list of all elevation values
        for x in [Elevation_Out_of_Bound]:
            self._populate(empty_masterdict, gridlist2, x, width, height)
        for x in [Bamboo, Coniferous, Broadleaf, Mixed, Lichen, Deciduous,
                  Shrublands, Clouds, Farmland, Outside_FNNR]:
            self._populate(empty_masterdict, gridlist, x, width, height)

        masterdict = empty_masterdict


        startinglist = masterdict['Broadleaf'] + masterdict['Mixed'] + masterdict['Deciduous']
        # Agents will start out in high-probability areas.
        for coordinate in masterdict['Elevation_Out_of_Bound'] + masterdict['Household'] + masterdict['PES'] \
                    + masterdict['Farm'] + masterdict['Forest']:
                if coordinate in startinglist:
                    startinglist.remove(coordinate)

        for line in _readCSV(hh_file)[1:]:  # see 'hh_survey.csv'
            hh_id_match = int(line[0])
            resource_name = line[1]  # frequency is monthly; currently not-used
            frequency = float(line[2]) / 6  # divided by 6 for 5-day frequency, as opposed to 30-day (1 month)
            y = int(line[5])
            x = int(line[6])
            resource = Resource(_readCSV(hh_file)[1:].index(line),
                                self, (x, y), hh_id_match, resource_name, frequency)
            self.grid.place_agent(resource, (int(x), int(y)))
            resource_dict.setdefault(hh_id_match, []).append(resource)


        # Create agents
        for i in range(self.num_agents):
            age = random.randint(1,5)
            wolf = WolfAgent(i, age, self)
            self.schedule.add(wolf)
            # add agents to grid
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            ind = random.randint(0,len(startinglist))
            # start = random.choice(startinglist)
            start = startinglist[ind]
            # self.grid.place_agent(a, (x,y))
            self.grid.place_agent(wolf, start)
        self.datacollector = DataCollector(
            agent_reporters={"Alive": "alive","Age": "age", "Pos": "pos"},
            model_reporters={"Pack Health": compute_pack_health}
        )

    def step(self):
        '''Advance the model by one step.'''
        self.time += 1
        self.datacollector.collect(self)
        self.schedule.step()

        # add other "events" here
        # radio towers do detection
        # GPS / Satellite do detection
        # helicopter / plane do tracking with detection information

    def _readASCII(self, text):
        # reads in a text file that determines the environmental grid setup from ABM
        f = open(text, 'r')
        body = f.readlines()
        width = body[0][-4:]  # last 4 characters of line that contains the 'width' value
        height = body[1][-5:]
        abody = body[6:]  # ASCII file with a header
        f.close()
        abody = reversed(abody)
        cells = []
        for line in abody:
            cells.append(line.split(" "))
        return [cells, int(width), int(height)]

    def _populate(self, masterdict, grid, land_type, width, height):
        # places land tiles on the grid - connects color/land cover category with ASCII file values from ABM
        counter = 0  # sets agent ID - not currently used
        for y in range(height):  # for each pixel,
            for x in range(width):
                value = float(grid[y][x])  # value from the ASCII file for that coordinate/pixel, e.g. 1550 elevation
                land_grid_coordinate = x, y
                land = land_type(counter, self)
                if land_type.__name__ == 'Elevation_Out_of_Bound':
                    if (value < land_type.lower_bound or value > land_type.upper_bound) and value != -9999:
                        # if elevation is not 1000-2200, but is within the bounds of the FNNR, mark as 'elevation OOB'
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                elif land_type.__name__ == 'Forest':
                    if land_type.type == value:
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                elif land_type.__name__ == 'PES':
                    if land_type.type == value:
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                elif land_type.__name__ == 'Farm':
                    if land_type.type == value:
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                elif land_type.__name__ == 'Household':
                    if land_type.type == value:
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                else:  # vegetation background
                    if land_type.type == value:
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1

