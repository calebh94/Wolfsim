import matplotlib.pyplot as plt
from mesa.batchrunner import BatchRunner
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from datetime import datetime
import numpy as np
from WolfsimModel import WolfModel, compute_pack_health, compute_pack_position, \
    compute_est_pack_position, agent_portrayal, home_range, compute_track_error, compute_track_error_average
from utils import readASCII


def sim_run(width, height, elev_cells, veg_cells, tracking_type, save=False):
    iterations = 1
    tf = 365*1
    wolfpack_size = 25
    all_ages = []
    for j in range(iterations):
        model = WolfModel(wolfpack_size, width, height, elev_cells, veg_cells,
                          tracking_type=tracking_type, plot_movement=False)
        for i in range(tf):
            model.step()
        for agent in model.schedule.agents:
            if agent.alive:
                all_ages.append(int(agent.age))
        # agent_age = [a.age for a in model.schedule.agents]
    plt.hist(((all_ages)), bins=range(max((all_ages))+1))
    plt.show()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    for wolf in model.schedule.agents:
        pos = wolf.pos
        agent_counts[pos[0]][pos[1]] += 1

    plt.imshow(agent_counts, interpolation='nearest')
    plt.colorbar()
    plt.show()

    # wolfpack_health = model.datacollector.get_model_vars_dataframe()
    # wolfpack_health.plot()
    wolfpack_data = model.datacollector.get_agent_vars_dataframe()
    wolfsim_data = model.datacollector.get_model_vars_dataframe()
    # wolfpack_data.plot()
    # Selecting data from final step of wolves
    startpoint = -wolfpack_size + 1
    last_set_all = wolfpack_data.iloc[startpoint:]
    last_set = last_set_all[last_set_all.Alive]
    print("Average wolf age at final step that is living: {}".format(np.average(last_set[last_set.Alive==True]["Age"])))
    print("Number of wolves that have died: {}".format(len(last_set[last_set.Alive==False])))
    # plt.show()
    # print(last_set)

    if save:
        time = datetime.now().strftime("%m_%d_%H_%M")
        wolfpack_data.to_csv("results/WolfSim_Results_{}.csv".format(time))

    kernel = True
    if kernel:
        home_range(wolfsim_data)


def batch_run(width, height, elev_cells, veg_cells):
    fixed_params = {"width": width, "height": height, "elev": elev_cells, "veg": veg_cells,
                    "N": 25}
    variable_params = {"N": range(10, 50, 1), "tracking_type": ['satellite', 'planes', 'helicopters', 'stations']}
    # variable_params = {"tracking_type": ['satellite', 'planes', 'helicopters', 'stations']}

    batch_run = BatchRunner(WolfModel,
                            variable_params,
                            fixed_params,
                            iterations=30,
                            max_steps=365,
                            model_reporters={"pack_health": compute_pack_health,
                                             "track_error": compute_track_error_average})
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    run_data.head()
    # plt.scatter(run_data.N, run_data.pack_health)
    plt.scatter(run_data.track_error, run_data.tracking_type)
    # plt.scatter(run_data.track_error, run_data.number_agents)
    plt.show()


def viz_run(width, height, elev_cells, veg_cells, tracking_type):
    grid = CanvasGrid(agent_portrayal, width, height, 1200, 900)
    chart = ChartModule([{"Label": "Pack Health",
                          "Color": "Black"}], canvas_height=200, canvas_width=500,
                        data_collector_name='datacollector')
    chart2 = ChartModule([{"Label":  "Pack Average Track Error",
                          "Color": "Black"}], canvas_height=200, canvas_width=500,
                        data_collector_name='datacollector')
    server = ModularServer(WolfModel,
                           [grid, chart, chart2],
                           "WolfSim",
                           {"N": 25, "width": width, "height": height,
                            "elev": elev_cells, "veg": veg_cells, "tracking_type": tracking_type})
    server.port = 8520  # The defaul
    server.launch()


if __name__ == "__main__":
    # file for input
    # vegetation_file_OLD = 'data/vegetation.txt'
    # vegetation_file = 'data/400x300TreeASCbyte2.asc'
    # elevation_file_OLD = 'data/DEM.txt'
    # elevation_file = 'data/400x300ElevASCint16.asc'
    vegetation_file = '200x150Tree.asc'
    elevation_file = '200x150Elev.asc'
    version = "new"

    # reading and prepping data for vegetation and elevation
    veg_cells, width, height = readASCII(vegetation_file, version=version)
    elev_cells, width, height = readASCII(elevation_file, version=version)

    # Select tracking method
    # tracking_type = "satellite"
    tracking_type = "planes"
    # tracking_type = "helicopters"
    # tracking_type = "stations"


    # Choose whether to run single analysis, batch analysis, or visual
    # sim_run(width, height, elev_cells,  veg_cells, tracking_type, save=False)
    # batch_run(width, height, elev_cells, veg_cells)
    viz_run(width, height, elev_cells, veg_cells, tracking_type)
