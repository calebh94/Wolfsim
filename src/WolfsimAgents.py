"""
Definition of and the functions for the Wolfsim Agents: Wolves, etc.
"""

from mesa import Agent
from numpy import random
import numpy as np

class WolfAgent(Agent):
    """An agent of a single wolf."""
    def __init__(self, unique_id, age, model):
        super().__init__(unique_id, model)
        self.alive = True
        self.age = age

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
            if random.randint(1,200) < 2:
                other.alive = False
                # print("Wolf " + str(self.unique_id) + " attacked and killed wolf " + str(other.unique_id))

    def step(self, debug=False):
        if random.randint(1,10000) < 2:
            self.alive = False
            # self.pos = None
        if not self.alive:
            return
        if self.age > 5:
            self.attack()
        if debug:
            print("Agent " + str(self.unique_id) + " with age " + str(self.age) +" is moving from position " + str(self.pos))
        self.age += 1/365
        self.move()

    def move_decision(self , options, target, avg_position, elevation_map):
        # current_position = self.neighbor_choice(options, masterdict)
        #TODO: utilize elevation map as cost in location
        current_position = self.pos
        likelihoods = [0.01, 0.55, 0.44] # [random, target, average]
        move_method = self.random.choices([0,1,2], weights=likelihoods)
        if move_method[0] == 0:
            new_position = self.random.choice(options)
        elif move_method[0] == 1:
            new_position = self.toward(options, target)
        elif move_method[0] == 2:
            new_position = self.toward(options, avg_position)
        else:
            new_position = self.random.choice(options)

        return new_position

    def toward(self, lists, target):
        new_ind = np.argmin(np.linalg.norm(np.array(lists) - np.array(target), axis=1))
        new_position = lists[new_ind]
        return new_position

    def check_elevation_of_neighbor(self, neighborlist, elevation_grid):
        #TODO: implement this for weights on elevation
        print("Coming Soon")
        raise RuntimeError

    def check_vegetation_of_neighbor(self, neighborlist, neighbordict):
        #TODO: implement this for weights on wwhat vegetation to go towards
        print("Coming Soon")
        raise RuntimeError