"""
Storage for util functions in WolfSim
"""

from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import scipy.stats as stats


class weibull_distribution():
    def __init__(self, n, a):
        self.n = n
        self.a = a

    def weib(self, x):
        a = self.a
        n = self.n
        return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)


def find_toward(lists, target):
    new_ind = np.argmin(np.linalg.norm(np.array(lists) - np.array(target), axis=1))
    new_position = lists[new_ind]
    return new_position


def compute_pack_health(model):
    # agent_ages = [agent.age for agent in model.schedule.agents if agent.alive == True]
    # avgage = np.average(agent_ages)
    alive_cnt = 0
    for agent in model.schedule.agents:
        if agent.alive:
            alive_cnt = alive_cnt + 1
    alive_percentage = alive_cnt / model.num_agents
    return alive_percentage


def compute_pack_position(model):
    agents_pos = [agent.pos for agent in model.schedule.agents if agent.alive == True]
    avg_pos = np.average(agents_pos,axis=0)
    return avg_pos


def compute_est_pack_position(model):
    agents_pos = [agent.pos for agent in model.schedule.agents if (agent.alive == True and agent.detected == True)]
    if len(agents_pos) < 1:
        avg_pos = model.tracked_position
    else:
        avg_pos = np.average(agents_pos,axis=0)
        model.tracked_position = avg_pos
    return avg_pos


def compute_track_error(model):
    true_pos = compute_pack_position(model)
    track_pos = compute_est_pack_position(model)
    distance = np.linalg.norm(np.array(true_pos) - np.array(track_pos))
    return distance


def compute_track_error_average(model):
    now_error = compute_track_error(model)
    model.track_error_sum += now_error
    track_error_average = model.track_error_sum / model.time
    return track_error_average



def compute_number_of_collars(model):
    collars = [agent.pos for agent in model.schedule.agents if agent.alive == True and (agent.collar_type == 1 or agent.collar_type ==2)]
    avg_pos = len(collars)
    return avg_pos


def compute_updated_target(model):
    agents_pos = [agent.pos for agent in model.schedule.agents if agent.alive]
    current_target = model.target
    distance = np.average(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_target]),axis=1))
    distance = np.linalg.norm(np.average(np.array(agents_pos), axis=0) - np.array(model.sites[current_target]))
    # distance = np.average(np.linalg.norm(np.array(agents_pos) - np.array(model.sites[current_target]),axis=1))
    if distance <= 20:
        model.feeding = model.feeding + 1
        if model.feeding >= 5:
            new_target = np.random.randint(0,len(model.sites)-1)
        else:
            new_target = current_target
    else:
        model.feeding = 0
        new_target = current_target
    return new_target


def compute_updated_target_pathing(model, plot=False):
    agents_pos = [agent.pos for agent in model.schedule.agents if agent.alive]
    current_waypoint = model.path[0]
    feeding_site = model.target
    distance = np.linalg.norm(np.average(np.array(agents_pos), axis=0) - np.array(current_waypoint))
    if distance <= 5.0 :
        if len(model.path) == 1: # reached target position
            model.feeding = model.feeding + 1
            if model.feeding >= 5:
                new_target = np.random.randint(0,len(model.sites)-1)
                source = np.average(np.array(agents_pos), axis=0).astype(np.int)
                source = tuple(source)
                target = model.sites[new_target]
                # Find shortest path for wolfpack travel using
                # new_path = nx.algorithms.shortest_paths.generic.shortest_path(model.G, source=source, target=target,
                #                                                       weight='weight')
                new_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(model.G, source=source, target=target,
                                                                               weight='weight')
                if plot:
                    x, y = zip(*new_path)
                    # print(y)
                    plt.scatter(y, x, marker='o')
                    plt.gca().invert_yaxis()
                    plt.imshow(model.tree, cmap='summer', interpolation='nearest', alpha=.5)
                    plt.imshow(model.grid_elevation, cmap='hot', interpolation='nearest')
                    plt.title("* PLEASE DO NOT CLOSE FIGURE* \n Wolfpack Optimal Paths")
                    # plt.show()
                    plt.pause(3)
            else:
                new_target = feeding_site
                new_path = model.path
        else:
            model.path.pop(0)
            new_path = model.path
            new_target = feeding_site
    else:
        model.feeding = 0
        new_target = feeding_site
        new_path = model.path

    return new_target, new_path


def plot_agent_positions(model, fig, ax):
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for wolf in model.schedule.agents:
        pos = wolf.pos
        agent_counts[pos[0]][pos[1]] += 1

    ax.imshow(agent_counts, interpolation='nearest')
    # ax.set_title("Wolf position at time {}".format(model.time))
    plt.title("Wolf position at time {}".format(model.time))
    # ax.colorbar()
    fig.savefig("results/wolf_positions_{}".format(model.time))
    # plt.show()
    # plt.pause(0.1)


def plot_feeding_zones(model, startinglist):
    feeding_zones = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    for pos in startinglist:
        feeding_zones[pos[0]][pos[1]] += 1

    plt.imshow(feeding_zones, interpolation='nearest')
    plt.title("Vegetation Map")
    plt.colorbar()
    plt.show()
    # plt.pause(0.1)


def readCSV(text):
    cells = []
    f = open(text, 'r')
    body = f.readlines()
    for line in body:
        cells.append(line.split(","))
    return cells


def readASCII(text, version):
    if version == "new":
        # reads in a text file that determines the environmental grid setup from ABM
        f = open(text, 'r')
        body = f.readlines()
        width = body[0][-4:-1]  # last 4 characters of line that contains the 'width' value
        height = body[1][-5:-1]
        abody = body[7:]  # ASCII file with a header
        f.close()
        abody = reversed(abody)
        cells = []
        for line in abody:
            cells.append(line.replace("\n","").split(" "))
        return [cells, int(width), int(height)]
    elif version == "old":
        # reads in a text file that determines the environmental grid setup from ABM
        f = open(text, 'r')
        body = f.readlines()
        width = body[0][-4:]  # last 4 characters of line that contains the 'width' value
        height = body[1][-5:]
        abody = body[6:]  # ASCII file with a header
        f.close()
        abody = reversed(abody)
        cells = []
        for line in abody:
            cells.append(line.split(" "))
        return [cells, int(width), int(height)]
    else:
        raise ValueError("Version should 'new' or 'old'")


def home_range(wolfsim_data):
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


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": .5,
                 "Layer": 0
                 }

    # if hasattr(agent, 'alive'):  # wolves
    if agent.type == 4:  # wolves
        if agent.alive and agent.detected:
            portrayal["Color"] = "red"
            portrayal["Layer"] = 9
            portrayal["r"] = 3
        elif agent.alive is False:
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 1
            portrayal["r"] = 3
        else:
            portrayal["Color"] = "black"
            portrayal["Layer"] = 1
            portrayal["r"] = 3
    elif agent.type == 5 and agent.active==False:  # detection agent
        portrayal["Color"] = "yellow"
        portrayal["Layer"] = 4
        portrayal["r"] = 5
    elif agent.type == 5 and agent.active==True:  # detection agent
        portrayal["Color"] = "yellow"
        portrayal["Layer"] = 5
        portrayal["r"] = agent.dist
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
            portrayal["Layer"] = 0
            portrayal["r"] = 10.0
            portrayal["Filled"] = "false"
        elif agent.type == 1:
            portrayal["Shape"] = "bin/map.png"
            portrayal["x"] = 600
            portrayal["y"] = 400
            portrayal["scale"] = 400
            portrayal["text"] = ""
            portrayal["text_color"] = "blue"
            portrayal["Filled"] = "false"
            portrayal["Layer"] = 0
        else:
            portrayal["Color"] = "white"
            portrayal["Layer"] = 0
            # portrayal["r"] = 1
            portrayal["Shape"] = "rect"
            portrayal["w"] = 1.0
            portrayal["h"] = 1.0
            portrayal["filled"] = "false"

    else:
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
    return portrayal
