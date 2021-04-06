from mesa import Agent, Model
from mesa.time import RandomActivation
from numpy import random
import matplotlib.pyplot as plt
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import numpy as np


def compute_pack_health(model):
    agent_ages = [agent.age for agent in model.schedule.agents]
    agent_health = [agent.alive for agent in model.schedule.agents]
    agesum = np.sum(agent_ages[agent_health==True]) / len(agent_health) #TODO: not calculccated right?
    return agesum


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
            agent_reporters={"Alive": "alive","Age": "age"},
            model_reporters={"Pack Health": compute_pack_health}
        )

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()


def sim_run():
    iterations = 1
    tf = 5
    wolfpack_size = 10
    width = 10
    height = 10
    all_ages = []
    for j in range(iterations):
        model = WolfModel(wolfpack_size, width, height)
        for i in range(tf):
            model.step()
        for agent in model.schedule.agents:
            all_ages.append(agent.age)
        # agent_age = [a.age for a in model.schedule.agents]
    plt.hist(all_ages, bins=range(max(all_ages)+1))
    plt.show()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    plt.imshow(agent_counts, interpolation='nearest')
    plt.colorbar()
    plt.show()

    wolfpack_health = model.datacollector.get_model_vars_dataframe()
    wolfpack_health.plot()
    plt.show()
    agent_age = model.datacollector.get_agent_vars_dataframe()
    print(agent_age.head())

    # alive_count = 0
    # for agent in model.schedule.agents:
    #     if agent.alive:
    #         alive_count += 1
    # print("Wolves remaining alive: " + str(alive_count) + "/" +str(wolfpack_size))

def batch_run():
    fixed_params = {"width": 10,
                    "height": 10}
    variable_params = {"N": range(10, 40, 10)}

    batch_run = BatchRunner(WolfModel,
                            variable_params,
                            fixed_params,
                            iterations=5,
                            max_steps=100,
                            model_reporters={"pack_health": compute_pack_health})
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    run_data.head()
    plt.scatter(run_data.N, run_data.pack_health)
    plt.show()


if __name__ == "__main__":
    # sim_run()
    batch_run()


