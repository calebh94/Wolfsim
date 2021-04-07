"""
Definition of and the functions for the Wolfsim Agents: Wolves, etc.
"""

from mesa import Agent
from numpy import random

class WolfAgent(Agent):
    """An agent of a single wolf."""
    def __init__(self, unique_id, age, model):
        super().__init__(unique_id, model)
        self.alive = True
        self.age = age

    def move(self):
        step_options = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)
        new_position = self.random.choice(step_options)
        self.model.grid.move_agent(self, new_position)

    def attack(self):
        neighbors = self.model.grid.get_cell_list_contents([self.pos])
        if len(neighbors) > 1:
            other = self.random.choice(neighbors)
            if random.randint(1,10) < 2:
                other.alive = False
                # print("Wolf " + str(self.unique_id) + " attacked and killed wolf " + str(other.unique_id))

    def step(self, debug=False):
        if random.randint(1,10) < 3:
            self.alive = False
        if not self.alive:
            return
        if self.age > 8:
            self.attack()
        if debug:
            print("Agent " + str(self.unique_id) + " with age " + str(self.age) +" is moving from position " + str(self.pos))
        self.age += 1
        self.move()