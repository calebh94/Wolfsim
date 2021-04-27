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

from WolfsimAgents import WolfAgent, DetectAgent
from environment import *
from utils import *

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


# Wolf simulation model class
class WolfModel(Model):
    """A model with some number of wolves."""
    # def __init__(self, N, width, height, plot_movement = False,tracking_type, n_collars):
    def __init__(self, N, width, height, elev, veg, tracking_type="planes", n_collars=1, plot_movement=False):
        super().__init__()
        self.num_agents = N
        # if OLD:
        #     width = self.readASCII_OLD(vegetation_file_OLD)[1] # width as listed at the beginning of the ASCII file
        #     height = self.readASCII_OLD(vegetation_file_OLD)[2] # height as listed at the beginning of the ASCII file
        # else:
        #     width = self.readASCII(vegetation_file)[1] # width as listed at the beginning of the ASCII file
        #     height = self.readASCII(vegetation_file)[2] # height as listed at the beginning of the ASCII file
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        # self.elev = elev
        self.grid_elevation = np.zeros((width, height))
        # self.grid_elevation = elev.copy()
        # self.veg = veg
        self.schedule = RandomActivation(self)
        self.running = True
        self.time = 0
        self.sites = None
        self.avg_pos = np.array([0,0])
        self.target = 0
        self.plot_movement = plot_movement
        self.feeding = 0
        self.path = []
        self.tracked_position = None
        self.tracking_type=tracking_type

        if tracking_type=="satellite":
            c_type=2
        else:
            c_type=1

        empty_masterdict = {'Outside_FNNR': [], 'Elevation_Out_of_Bound': [], 'Vegetation': []}

        # TODO: modify to only do this on the first run when running batch cases
        # if OLD:
        #     tree = self.readASCII_OLD(vegetation_file_OLD)[0]  # list of all coordinate values; see readASCII function
        #     elev = self.readASCII_OLD(elevation_file_OLD)[0]  # list of all elevation values
        # else:
        #     tree = self.readASCII(vegetation_file)[0]  # list of all coordinate values; see readASCII function
        #     elev = self.readASCII(elevation_file)[0]  # list of all elevation values
        self.elevation_fill(elev)

        for x in [Elevation_Out_of_Bound]: #TODO: we need to use this info to restrict movement through "obstacles"
            self.populate(empty_masterdict, elev, x, width, height)
        for x in [Vegetation, Outside_FNNR]:
            self.populate(empty_masterdict, veg, x, width, height)
        # adding image agent in
        # img_agent = Img(0, self)
        # self.grid.place_agent(img_agent, (0,0))

        # graph structure for path planning
        # tree = np.array(veg).astype(np.int).transpose()
        tree = np.zeros((height, width))
        for i in range(len(veg)):
            for j in range(len(veg[0])):
                tree[i,j] = int(veg[i][j])
        tree = tree.transpose()
        self.tree = tree.copy()


        # self.tree = tree.copy()
        elevation = self.grid_elevation.copy()
        # elevation = elevation.transpose()

        self.G = nx.generators.lattice.grid_2d_graph(self.width, self.height, periodic=False)
        tree_value = 1 * 10
        elev_value = 0.1 * 30
        for e in self.G.edges():
            # print(Tree[e[0]])
            if elevation[e[0]] + elevation[e[1]] < 2 or tree[e[0]] + tree[e[1]] > 0 :
                self.G[e[0]][e[1]]['weight'] = 99999999
            else:

                self.G[e[0]][e[1]]['weight'] = tree_value * (abs(tree[e[0]] + tree[e[1]] - 4)) + elev_value * (
                        abs((elevation[e[0]] - elevation[e[1]])) / 2)

        masterdict = empty_masterdict

        startinglist = masterdict['Vegetation']
        # Agents will start out in high-probability areas.
        for coordinate in masterdict['Elevation_Out_of_Bound']:
                if coordinate in startinglist:
                    startinglist.remove(coordinate)

        # self.sites = startinglist  # the target locations are formed from the startinglist above
        num_sites = 8
        site_indices = np.linspace(0,len(startinglist)-1, num_sites)
        self.sites = [startinglist[int(ind)] for ind in site_indices]
        self.target = self.random.randint(0,num_sites-1) # update
        # the initial target to a random location
        # self.target=0
        self.path = [self.sites[self.target]]  # initialize the wolf path to a list of only the intial location
        # self.target, self.path = compute_updated_target_pathing(self, plot=False)
        self.tracked_position = self.sites[self.target]

        for p in range(0,len(self.sites)):
            feed_site_agent = Vegetation(p, self, pos=self.sites[p], elevation=self.grid_elevation[self.sites[p]])
            self.grid.place_agent(feed_site_agent, self.sites[p])

        plot_sites = False
        if plot_sites:
            plot_feeding_zones(self)

        # Create agents
        for i in range(self.num_agents):
            age = random.randint(1,5)
            if i <n_collars:
                collar=c_type
            else:
                collar=0
            wolf = WolfAgent(i, age, collar, self)
            self.schedule.add(wolf)
            # add agents to grid
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            # ind = random.randint(0,len(startinglist))
            # start = random.choice(startinglist)
            # start = startinglist[ind]
            start = startinglist[self.target]
            # start = tuple(start + random.randint(0,5,2)) # randomly push around target location
            # self.grid.place_agent(a, (x,y))
            self.grid.place_agent(wolf, start)

        # Create Trackers
        if self.tracking_type=="satellite":
            sat_pos = (100,100) #TODO: why does this break
            tracker = DetectAgent(i+self.num_agents, "satellite", sat_pos, self)
            self.schedule.add(tracker)
            self.grid.place_agent(tracker, tracker.pos)
        if self.tracking_type=="stations":
            stat_pos = [(60,100), (160,100)] #TODO: why does this break
            for i in range(len(stat_pos)):
                tracker = DetectAgent(i+self.num_agents, self.tracking_type, stat_pos[i], self)
                self.schedule.add(tracker)
                self.grid.place_agent(tracker, tracker.pos)
        if self.tracking_type=="helicopters":
            stat_pos = self.sites #TODO: why does this break
            for i in range(len(stat_pos)):
                tracker = DetectAgent(i+self.num_agents, self.tracking_type, stat_pos[i], self)
                self.schedule.add(tracker)
                self.grid.place_agent(tracker, tracker.pos)
        if self.tracking_type == "planes":
            stat_pos = [(100, 50),(0, 100), (50,75),(116,100),(133,149),(150,35),(199,20)]  # TODO: why does this break
            for i in range(len(stat_pos)):
                tracker = DetectAgent(i + self.num_agents, self.tracking_type, stat_pos[i], self)
                self.schedule.add(tracker)
                self.grid.place_agent(tracker, tracker.pos)

        self.datacollector = DataCollector(
        agent_reporters={"Alive": "alive","Age": "age", "Pos": "pos"},
        model_reporters={"Pack Health": compute_pack_health ,"Pack Position": compute_pack_position, "Pack Estimated Position": compute_est_pack_position,
                             "Pack Track Error": compute_track_error}
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
        self.tracked_position = compute_est_pack_position(self)
        # self.target = compute_updated_target(self)
        self.target, self.path = compute_updated_target_pathing(self, plot=False)
        if self.plot_movement and self.time % 25 == 0:
            plot_agent_positions(self, fig, ax)
        for wolfv in self.schedule.agents:
            if wolfv.type==4:
                wolfv.detected=False
            if wolfv.type==5:
                wolfv.active=False
        self.schedule.step()

    def elevation_fill(self, grid):
        for y in range(self.height):  # for each pixel,
            for x in range(1,self.width):
                value = float(grid[y][x])  # val
                self.grid_elevation[x][y] = value

        plot=False
        if plot:
            # np.save("elevation_image.)
            plt.imshow(self.grid_elevation)
            plt.clim(0, self.grid_elevation.max())
            plt.colorbar()
            plt.title("Elevation Map")
            plt.show()

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
                        # self.grid.place_agent(land, land_grid_coordinate)
                        masterdict[land.__class__.__name__].append(land_grid_coordinate)
                        counter += 1
                else:  # elevation background
                # if land_type.type == value:
                    land.elevation = value
                    # self.grid.place_agent(land, land_grid_coordinate)
                    masterdict[land.__class__.__name__].append(land_grid_coordinate)
                    counter += 1

