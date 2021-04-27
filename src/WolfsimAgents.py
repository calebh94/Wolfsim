"""
Definition of and the functions for the Wolfsim Agents: Wolves, etc.
"""

from mesa import Agent
from numpy import random
import numpy as np
from utils import weibull_distribution, find_toward


class WolfAgent(Agent):
    """An agent of a single wolf."""
    def __init__(self, unique_id, age, collar_type, model):
        super().__init__(unique_id, model)
        self.alive = True
        self.age = age
        self.pos = None
        self.collar_type = collar_type  # Collar Types[1: radio, 2:GPS]
        self.collar_health = 100
        self.detected = False
        self.tracked_pos = None
        self.type = 4

        self.collar1_dist = weibull_distribution(1, 0.8)
        self.collar2_dist = weibull_distribution(1, 1)


    def move(self):
        if self.pos is None:
            return
        step_options = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)
        site_options = self.model.get_sites()
        target_ind = self.model.target
        # current_target = site_options[target_ind]
        current_target = self.model.path[0]

        new_position = self.move_decision(step_options, current_target, self.model.avg_pos, self.model.grid_elevation) #TODO: change last value to pack avergage pos
        # new_position = self.random.choice(step_options)
        self.model.grid.move_agent(self, new_position)

    def attack(self):
        neighbors = self.model.grid.get_cell_list_contents([self.pos])
        if len(neighbors) > 1:
            other = self.random.choice(neighbors)
            if random.randint(1,5000) < 2:
                other.alive = False

    def step(self, debug=False):
        # Chance that wolf is lost to health problems
        if random.randint(1,10000) < 1:
            self.alive = False
            # self.pos = None
        if not self.alive:
            return
        # Chance that wolf attacks nearby wolves after age reaches greater than 5 (territorial)
        if self.age > 5:
            self.attack()
        if debug:
            print("Agent " + str(self.unique_id) + " with age " + str(self.age) +" is moving from position " + str(self.pos))
        # Increase age by 1 day
        self.age += 1/365
        # TODO: tracking related things
        if self.collar_health > 1:  # failure at 1%
            #a=1  # Do things here
            #if np.random.randint(0,100) > 50:
                #self.detected = True
                #self.tracked_pos = tuple(self.pos + random.randint(0,5,2))
            #else:
                #self.detected = False  # temp since not implemented yet
            # collar wear and tear
            if self.collar_type == 1:  # radio collar
                # damage = 5
                # self.collar_health = self.collar_health - damage
                self.collar_health = self.collar1_dist.weib(self.model.time/(365))*100
            elif self.collar_type == 2:
                damage = 1
                self.collar_health = self.collar_health - damage
                self.collar_health = self.collar2_dist.weib(self.model.time/(365))*100
        # Move agent
        self.move()

    def move_decision(self , options, target, avg_position, elevation_map):
        likelihoods = [0.01, 0.55, 0.44]  # [random, target, average]
        move_method = self.random.choices([0,1,2], weights=likelihoods)
        if move_method[0] == 0:
            new_position = self.random.choice(options)
        elif move_method[0] == 1:
            new_position = find_toward(options, target)
        elif move_method[0] == 2:
            new_position = find_toward(options, avg_position)
        else:
            new_position = self.random.choice(options)
        return new_position


class DetectAgent(Agent):
    """An agent of detection."""

    def __init__(self, unique_id, tracking_type, pos, model):
        super().__init__(unique_id, model)
        self.active = False
        self.alive = False
        #TODO: add agent type
        self.age=99
        self.pos=pos
        self.type = 5
        self.tracking_type=tracking_type
        if self.tracking_type=="satellite":
            self.timer=5
            self.dist=1000
        if self.tracking_type=="stations":
            self.timer=5
            self.dist=100
        if self.tracking_type=="helicopters":
            self.timer=5
            self.dist=75
        if self.tracking_type=="planes":
            self.timer=5
            self.dist=100


    def step(self, debug=False):
        if self.model.time % (self.timer) == 0 and not(self.tracking_type== "helicopters"  or self.tracking_type== "planes"):
            self.active = True
            self.detect_nearby()
        elif self.model.time % (self.timer) == 0 and self.tracking_type== "helicopters":
            if self.model.time / (self.timer) % 4 == 1:
                if self.unique_id-self.model.num_agents<2:
                    self.active=True
                    self.detect_nearby()
            if self.model.time / (self.timer) % 4 == 2:
                if self.unique_id-self.model.num_agents<4 and self.unique_id-self.model.num_agents>1:
                    self.active=True
                    self.detect_nearby()
            if self.model.time / (self.timer) % 4 == 3:
                if self.unique_id-self.model.num_agents<6 and self.unique_id-self.model.num_agents>3:
                    self.active=True
                    self.detect_nearby()
            if self.model.time / (self.timer) % 4 == 0:
                if self.unique_id-self.model.num_agents<8 and self.unique_id-self.model.num_agents>5:
                    self.active=True
        elif self.model.time % (self.timer) == 0 and self.tracking_type== "planes":

            if self.model.time / (self.timer) % 3 == 1:
                if (self.unique_id-self.model.num_agents<3 and self.unique_id-self.model.num_agents>0) \
                        or self.unique_id-self.model.num_agents==0:
                    self.active=True
                    self.detect_nearby()
            if self.model.time / (self.timer) % 3 == 2:
                if (self.unique_id - self.model.num_agents < 5 and self.unique_id - self.model.num_agents > 2) \
                        or self.unique_id - self.model.num_agents == 0:
                    self.active=True
                    self.detect_nearby()
            if self.model.time / (self.timer) % 3 == 0:
                if (self.unique_id - self.model.num_agents < 7 and self.unique_id - self.model.num_agents > 4) \
                        or self.unique_id - self.model.num_agents == 0:
                    self.active=True
                    self.detect_nearby()

            #TODO: JJJ add set to no longer active for all detect

    def detect_nearby(self):
        for wolf in self.model.schedule.agents:
            if wolf.alive==True and (wolf.collar_type ==1 or wolf.collar_type ==2):
                if np.linalg.norm(np.array(wolf.pos)-np.array(self.pos)) < self.dist:
                    if self.tracking_type=="satellite":
                        if 25>random.randint(0,100):
                            wolf.detected=True
                    else:
                        if (self.dist-(np.linalg.norm(np.array(wolf.pos)-np.array(self.pos)))/self.dist*100) >random.randint(0,100):
                            wolf.detected = True