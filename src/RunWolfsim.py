from mesa import Agent, Model
from mesa.time import RandomActivation
from numpy import random
import matplotlib.pyplot as plt
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import numpy as np

from WolfsimModel import WolfModel, compute_pack_health

def sim_run():
    iterations = 1
    tf = 5
    wolfpack_size = 1000
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
    sim_run()
    # batch_run()
