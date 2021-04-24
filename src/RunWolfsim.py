import matplotlib.pyplot as plt
from mesa.batchrunner import BatchRunner
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np
import scipy.stats as stats
from WolfsimModel import WolfModel, compute_pack_health, compute_pack_position


def sim_run():
    iterations = 1
    tf = 1000
    wolfpack_size = 50
    width = 10 # using vegetation file
    height = 10 # using vegetation file
    all_ages = []
    for j in range(iterations):
        model = WolfModel(wolfpack_size, width, height, plot_movement=False)
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
    # wolfpack_data.plot()
    startpoint = -wolfpack_size + 1
    last_set = wolfpack_data.iloc[startpoint:]
    print("Average wolf age at final step that is living: {}".format(np.average(last_set[last_set.Alive==True]["Age"])))
    print("Number of wolves that have died: {}".format(len(last_set[last_set.Alive==False])))
    # plt.show()
    # print(last_set)

    kernel = False
    # DOING THE KDE
    if kernel:
        positions = wolfpack_data.Pos
        vals, cnts = np.unique(positions, return_counts=True)
        points = np.zeros((len(positions),2))
        for i in range(0,len(positions)):
            points[i][0] = positions.iloc[i][0]
            points[i][1] = positions.iloc[i][1]
        # kde = stats.gaussian_kde(points.transpose(), bw_method=None, weights=cnts, )
        kde = stats.gaussian_kde(points.transpose(), bw_method=None)

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
        ax.plot(points[:,0], points[:,1], 'k.', markersize=5)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.show()


def batch_run():
    fixed_params = {"width": 10,
                    "height": 10}
    variable_params = {"N": range(1, 40, 1)}

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


def viz_run():
    bounds = [0, 500, 1000, 2000, 5000]
    colors = ["orange", "red", "pink", "purple", "purple"]

    def agent_portrayal(agent):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "r": .5
                     }

        if hasattr(agent, 'alive'):  # wolves
            if agent.alive:
                portrayal["Color"] = "red"
                portrayal["Layer"] = 1
                portrayal["r"] = 10
            else:
                portrayal["Color"] = "blue"
                portrayal["Layer"] = 1
                portrayal["r"] = 10
        elif hasattr(agent, "elevation"):  # elevation
            # if agent.type == -9999:  # not available
            if agent.type == 99:
                portrayal["Color"] = "black"
                portrayal["Layer"] = 0
                # portrayal["r"] = 1
                portrayal["Shape"] = "rect"
                portrayal["w"] = 1.0
                portrayal["h"] = 1.0
            elif agent.type == 0:
                portrayal["Color"] = "green"
                portrayal["Layer"] = 1
                portrayal["r"] = 1.0
            else:
                portrayal["Color"] = "white"
                portrayal["Layer"] = 0
                # portrayal["r"] = 1
                portrayal["Shape"] = "rect"
                portrayal["w"] = 1.0
                portrayal["h"] = 1.0
                portrayal["filled"] = "false"
                #     diffs = np.array(bounds) - agent.elevation
                #     diffs[diffs < 0] = 10000
                #     index = np.argmin(diffs)
                #     color = colors[index]
                #     portrayal["Color"] = color
                #     portrayal["Layer"] = 0
                #     # portrayal["r"] = 1
                #     portrayal["Shape"] = "rect"
                #     portrayal["w"] = 1.0
                #     portrayal["h"] = 1.0

        else:
            portrayal["Color"] = "grey"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
        return portrayal

    # grid = CanvasGrid(agent_portrayal, 80, 100, 500, 500)
    grid = CanvasGrid(agent_portrayal, 400, 300, 1200, 900)

    server = ModularServer(WolfModel,
                           [grid],
                           "WolfSim",
                           {"N": 10, "width": 400, "height": 300})
    server.port = 8520  # The default
    server.launch()


if __name__ == "__main__":
    # Choose whether to run single analysis, batch analysis, or visual
    # sim_run()
    # batch_run()
    viz_run()
