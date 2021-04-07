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


def compute_pack_health(model):
    agent_ages = [agent.age for agent in model.schedule.agents]
    agent_health = [agent.alive for agent in model.schedule.agents]

    agesum = np.sum(agent_ages[agent_health==True]) / len(agent_health) #TODO: not calculccated right?
    return agesum


class WolfModel(Model):
    """A model with some number of wolves."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            age = random.randint(1,5)
            a = WolfAgent(i, age, self)
            self.schedule.add(a)
            # add agents to grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))

        self.datacollector = DataCollector(
            agent_reporters={"Alive": "alive","Age": "age", "Pos": "pos"},
            model_reporters={"Pack Health": compute_pack_health}
        )

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()
        # add other "events" here
        # radio towers do detection
        # GPS / Satellite do detection
        # helicopter / plane do tracking with detection information


