import matplotlib.pyplot as plt
from mesa.batchrunner import BatchRunner
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from datetime import datetime
import numpy as np
import scipy.stats as stats
from WolfsimModel import WolfModel, compute_pack_health, compute_pack_position, compute_est_pack_position, agent_portrayal
from utils import readASCII


def sim_run(width, height, elev_cells, veg_cells, tracking_type, save=False):
    iterations = 1
    tf = 500
    wolfpack_size = 50
    all_ages = []
    for j in range(iterations):
        model = WolfModel(wolfpack_size, width, height, elev_cells, veg_cells,
                          tracking_type=tracking_type, plot_movement=False)
        for i in range(tf):
            model.step()
        for agent in model.schedule.agents:
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
    startpoint = -wolfpack_size + 1
    last_set = wolfpack_data.iloc[startpoint:]
    print("Average wolf age at final step that is living: {}".format(np.average(last_set[last_set.Alive==True]["Age"])))
    print("Number of wolves that have died: {}".format(len(last_set[last_set.Alive==False])))
    # plt.show()
    # print(last_set)

    if save:
        time = datetime.now().strftime("%m_%d_%H_%M")
        wolfpack_data.to_csv("results/WolfSim_Results_{}.csv".format(time))

    kernel = False
    # DOING THE KDE
    if kernel:
        positions = wolfsim_data['Pack Position']
        positions_tracked = wolfsim_data['Pack Estimated Position']
        # vals, cnts = np.unique(positions, return_counts=True)
        points = np.zeros((len(positions),2))
        points_tracked = np.zeros((len(positions),2))
        for i in range(0,len(positions)):
            points[i][0] = positions.iloc[i][0]
            points[i][1] = positions.iloc[i][1]
            points_tracked[i][0] = positions_tracked.iloc[i][0]
            points_tracked[i][1] = positions_tracked.iloc[i][1]
        # kde = stats.gaussian_kde(points.transpose(), bw_method=None, weights=cnts, )
        kde = stats.gaussian_kde(points.transpose(), bw_method=None)
        kde_tracked = stats.gaussian_kde(points_tracked.transpose(), bw_method=None)

        mins = np.min(points,axis=0)
        xmin = mins[0]
        ymin = mins[1]
        maxs = np.max(points,axis=0)
        xmax = maxs[0]
        ymax = maxs[1]

        X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        testpts = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(testpts), X.shape)

        fig, ax = plt.subplots()
        ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                  extent=[xmin, xmax, ymin, ymax])
        ax.plot(points[:,0], points[:,1], 'k.', markersize=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        # plt.show()

        mins2 = np.min(points_tracked,axis=0)
        xmin2 = mins2[0]
        ymin2 = mins2[1]
        maxs2 = np.max(points_tracked,axis=0)
        xmax2 = maxs2[0]
        ymax2 = maxs2[1]

        X2,Y2 = np.mgrid[xmin2:xmax2:100j, ymin2:ymax2:100j]
        testpts2 = np.vstack([X2.ravel(), Y2.ravel()])
        Z_tracked = np.reshape(kde_tracked(testpts2), X.shape)
        fig2, ax2 = plt.subplots()
        ax2.imshow(np.rot90(Z_tracked), cmap=plt.cm.gist_earth_r,
                  extent=[xmin2, xmax2, ymin2, ymax2])
        ax2.plot(points_tracked[:,0], points_tracked[:,1], 'k.', markersize=2)
        ax2.set_xlim([xmin2, xmax2])
        ax2.set_ylim([ymin2, ymax2])
        plt.show()


def batch_run(width, height, elev_cells, veg_cells):
    fixed_params = {"width": width, "height": height, "elev": elev_cells, "veg": veg_cells}
    variable_params = {"N": range(1, 40, 1), "tracking_type": ['satellite', 'planes', 'helicopters', 'stations']}

    batch_run = BatchRunner(WolfModel,
                            variable_params,
                            fixed_params,
                            iterations=5,
                            max_steps=100,
                            model_reporters={"pack_health": compute_pack_health, "pack_position": compute_pack_position})
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    run_data.head()
    # plt.scatter(run_data.N, run_data.pack_health)
    plt.scatter(run_data.N, run_data.pack_health)
    plt.show()


def viz_run(width, height, elev_cells, veg_cells, tracking_type):
    grid = CanvasGrid(agent_portrayal, width, height, 1200, 900)
    chart = ChartModule([{"Label": "Pack Health",
                          "Color": "Black"}], canvas_height=200, canvas_width=500,
                        data_collector_name='datacollector')
    chart2 = ChartModule([{"Label": "Pack Track Error",
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
    vegetation_file = 'data/200x150Tree.asc'
    elevation_file = 'data/200x150Elev.asc'
    version = "new"

    # reading and prepping data for vegetation and elevation
    veg_cells, width, height = readASCII(vegetation_file, version=version)
    elev_cells, width, height = readASCII(elevation_file, version=version)

    # Select tracking method
    tracking_type = "satellite"
    # tracking_type = "planes"
    # tracking_type = "helicopters"
    # tracking_type = "stations"


    # Choose whether to run single analysis, batch analysis, or visual
    # sim_run(width, height, elev_cells,  veg_cells, tracking_type, save=False)
    # batch_run(width, height, elev_cells, veg_cells)
    viz_run(width, height, elev_cells, veg_cells, tracking_type)
