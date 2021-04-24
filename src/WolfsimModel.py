"""
Wolfsim Model using the Mesa Agent-based Python Framework

Wolfsim Model defines the agent-based simulation components such as the creation of the agents, form of the grid, etc.

"""

from mesa import Model
from mesa.time import RandomActivation
from numpy import random
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from WolfsimAgents import WolfAgent
from environment import *

# Plotting preparation
fig, ax = plt.subplots()
# DATA FILES
vegetation_file_OLD = 'data/vegetation.txt'
vegetation_file = 'data/400x300TreeASCbyte2.asc'
elevation_file_OLD = 'data/DEM.txt'
elevation_file = 'data/400x300ElevASCint16.asc'
OLD = False
# hh_file = 'data/hh_survey.csv'
resource_dict = {}

# Data and plotting functions
def compute_pack_health(model):
    # agent_ages = [agent.age for agent in model.schedule.agents]
    # agent_health = [agent.alive for agent in model.schedule.agents if agent.alive == True]
    #
    # agesum = np.sum(agent_ages[agent_health==True]) / len(agent_health)
    agent_ages = [agent.age for agent in model.schedule.agents if agent.alive == True
                  ]
    avgage = np.average(agent_ages)
    return avgage

def compute_pack_position(model):
    agents_pos = [agent.pos for agent in model.schedule.agents if agent.alive == True]
    avg_pos = np.average(agents_pos,axis=0)
    return avg_pos

def compute_updated_target(model):
    agents_pos = [agent.pos for agent in model.schedule.agents]
    current_target = model.target
    distance = np.average(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_target]),axis=1))
    # distance = np.average(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_target]),axis=1))
    if distance <= 10:
        model.feeding = model.feeding + 1
        if model.feeding >= 5:
            new_target = random.randint(0,len(model.sites)-1)
        else:
            new_target = current_target
    else:
        model.feeding = 0
        new_target = current_target
    return new_target


def compute_updated_target_pathing(model, plot=False):
    agents_pos = [agent.pos for agent in model.schedule.agents]
    current_waypoint = model.path[0]
    feeding_site = model.target
    # distance = np.average(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_waypoint]),axis=1))
    distance = np.average(np.linalg.norm(np.array(agents_pos) -np.array(current_waypoint),axis=1))
    # distance = np.max(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_target]),axis=1))
    if distance <= 7.5 :
        if len(model.path) == 1: # reached target position
            model.feeding = model.feeding + 1
            if model.feeding >= 5:
                new_target = random.randint(0,len(model.sites)-1)
                source = np.average(np.array(agents_pos), axis=0).astype(np.int)
                source = tuple(source)
                target = model.sites[new_target]
                # Find shortest path for wolfpack travel using
                # new_path = nx.algorithms.shortest_paths.generic.shortest_path(model.G, source=source, target=target,
                #                                                       weight='weight')
                new_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(model.G, source=source, target=target,
                                                                               weight='weight')
                if plot:
                    x, y = zip(*new_path)
                    print(y)
                    plt.scatter(y, x, marker='o')
                    plt.gca().invert_yaxis()
                    plt.imshow(model.tree, cmap='summer', interpolation='nearest', alpha=.5)
                    plt.imshow(model.grid_elevation, cmap='hot', interpolation='nearest')
                    plt.show()
            else:
                new_target = feeding_site
                new_path = model.path
        else:
            model.path.pop(0)
            new_path = model.path
            new_target = feeding_site
    else:
        model.feeding = 0
        new_target = feeding_site
        new_path = model.path

    return new_target, new_path

def plot_agent_positions(model):
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    for wolf in model.schedule.agents:
        pos = wolf.pos
        agent_counts[pos[0]][pos[1]] += 1

    ax.imshow(agent_counts, interpolation='nearest')
    # ax.set_title("Wolf position at time {}".format(model.time))
    plt.title("Wolf position at time {}".format(model.time))
    # ax.colorbar()
    fig.savefig("results/wolf_positions_{}".format(model.time))
    # plt.show()
    # plt.pause(0.1)

def plot_feeding_zones(model):
    feeding_zones = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    for pos in model.sites:
        feeding_zones[pos[0]][pos[1]] += 1

    plt.imshow(feeding_zones, interpolation='nearest')
    plt.title("Feeding sites")
    plt.colorbar()
    plt.show()
    # plt.pause(0.1)

def readCSV(text):
    cells = []
    f = open(text, 'r')
    body = f.readlines()
    for line in body:
        cells.append(line.split(","))
    return cells

# Wolf simulation model class
class WolfModel(Model):
    """A model with some number of wolves."""
    def __init__(self, N, width, height, plot_movement = False):
        super().__init__()
        self.num_agents = N
        if OLD:
            width = self.readASCII_OLD(vegetation_file_OLD)[1] # width as listed at the beginning of the ASCII file
            height = self.readASCII_OLD(vegetation_file_OLD)[2] # height as listed at the beginning of the ASCII file
        else:
            width = self.readASCII(vegetation_file)[1] # width as listed at the beginning of the ASCII file
            height = self.readASCII(vegetation_file)[2] # height as listed at the beginning of the ASCII file
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.grid_elevation = np.zeros((width, height))
        self.schedule = RandomActivation(self)
        self.running = True
        self.time = 0
        self.sites = None
        self.avg_pos = np.array([0,0])
        self.target = 0
        self.plot_movement = plot_movement
        self.feeding = 0
        self.path = []

        empty_masterdict = {'Outside_FNNR': [], 'Elevation_Out_of_Bound': [], 'Vegetation': []}

        # TODO: modify to only do this on the first run when running batch cases
        if OLD:
            tree = self.readASCII_OLD(vegetation_file_OLD)[0]  # list of all coordinate values; see readASCII function
            elev = self.readASCII_OLD(elevation_file_OLD)[0]  # list of all elevation values
        else:
            tree = self.readASCII(vegetation_file)[0]  # list of all coordinate values; see readASCII function
            elev = self.readASCII(elevation_file)[0]  # list of all elevation values
        self.elevation_fill(elev)

        for x in [Elevation_Out_of_Bound]: #TODO: we need to use this info to restrict movement through "obstacles"
            self.populate(empty_masterdict, elev, x, width, height)
        for x in [Vegetation, Outside_FNNR]:
            self.populate(empty_masterdict, tree, x, width, height)

        #TODO: need to do some optimizing of variables and objects around here***

        # graph structure for path planning
        tree = np.array(tree).astype(np.int).transpose()
        self.tree = tree.copy()
        elev = self.grid_elevation.copy()

        self.G = nx.generators.lattice.grid_2d_graph(self.width, self.height, periodic=False)
        tree_value = 1 * 10
        elev_value = 0.1 * 30
        for e in self.G.edges():
            # print(Tree[e[0]])
            if elev[e[0]] + elev[e[1]] < 2 or tree[e[0]] + tree[e[1]] < 2 :
                self.G[e[0]][e[1]]['weight'] = 99999999
            else:

                self.G[e[0]][e[1]]['weight'] = tree_value * (abs(tree[e[0]] + tree[e[1]] - 4)) + elev_value * (
                        abs((elev[e[0]] - elev[e[1]])) / 2)

        masterdict = empty_masterdict

        startinglist = masterdict['Vegetation']
        # Agents will start out in high-probability areas.
        for coordinate in masterdict['Elevation_Out_of_Bound']:
                if coordinate in startinglist:
                    startinglist.remove(coordinate)

        self.sites = startinglist  # the target locations are formed from the startinglist above
        self.target = self.random.randint(0,len(self.sites)) # update the initial target to a random location
        self.path = [self.sites[self.target]]  # initialize the wolf path to a list of only the intial location

        plot_sites = False
        if plot_sites:
            plot_feeding_zones(self)

        # Create agents
        for i in range(self.num_agents):
            age = random.randint(1,5)
            wolf = WolfAgent(i, age, self)
            self.schedule.add(wolf)
            # add agents to grid
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            # ind = random.randint(0,len(startinglist))
            # start = random.choice(startinglist)
            # start = startinglist[ind]
            start = startinglist[self.target]
            start = tuple(start + random.randint(0,5,2)) # randomly push around target location
            # self.grid.place_agent(a, (x,y))
            self.grid.place_agent(wolf, start)
        self.datacollector = DataCollector(
            agent_reporters={"Alive": "alive","Age": "age", "Pos": "pos"},
            model_reporters={"Pack Health": compute_pack_health ,"Pack Position": compute_pack_position}
        )

    def get_sites(self):
        if self.sites is not None:
            return self.sites
        else:
            raise ValueError

    def step(self):
        '''Advance the model by one step.'''
        self.time += 1 # day metric
        self.datacollector.collect(self)
        self.avg_pos = compute_pack_position(self)
        # self.target = compute_updated_target(self)
        self.target, self.path = compute_updated_target_pathing(self, plot=False)
        if self.plot_movement and self.time % 25 == 0:
            plot_agent_positions(self)
        self.schedule.step()
        # add other "events" here inside a Priority Queue
        # radio towers do detection
        # GPS / Satellite do detection
        # helicopter / plane do tracking with detection information

    def elevation_fill(self, grid):
        for y in range(self.height):  # for each pixel,
            for x in range(1,self.width):
                value = float(grid[y][x])  # val
                self.grid_elevation[x][y] = value

        plot=False
        if plot:
            plt.imshow(self.grid_elevation)
            plt.clim(0, self.grid_elevation.max())
            plt.colorbar()
            plt.title("Elevation Map")
            plt.show()

    def readASCII(self, text):
        # reads in a text file that determines the environmental grid setup from ABM
        f = open(text, 'r')
        body = f.readlines()
        width = body[0][-4:-1]  # last 4 characters of line that contains the 'width' value
        height = body[1][-5:-1]
        abody = body[7:]  # ASCII file with a header
        f.close()
        abody = reversed(abody)
        cells = []
        for line in abody:
            cells.append(line.replace("\n","").split(" "))
        return [cells, int(width), int(height)]

    def readASCII_OLD(self, text):
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

    def populate(self, masterdict, grid, land_type, width, height):
        counter = 0  # sets agent ID - not currently used
        for y in range(height):  # for each pixel,
            for x in range(1,width):
                value = float(grid[y][x])  # value from the ASCII file for that coordinate/pixel, e.g. 1550 elevation
                land_grid_coordinate = x, y
                land = land_type(counter, self)
                if land_type.__name__ == 'Elevation_Out_of_Bound':
                    if (value < land_type.lower_bound or value > land_type.upper_bound) and value != -9999:
                        # if elevation is not bounds but is within the bounds of the FNNR, mark as 'elevation OOB'
                        land.elevation = value
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                elif land_type.__name__ == 'Vegetation':
                    if land_type.type == value:
                        land.elevation = value
                        self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                else:  # elevation background
                # if land_type.type == value:
                    land.elevation = value
                    self.grid.place_agent(land, land_grid_coordinate)
                    masterdict[land.__class__.__name__].append(land_grid_coordinate)
                    counter += 1

