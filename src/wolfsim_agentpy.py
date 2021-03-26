# Model design
import agentpy as ap
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import IPython

def normalize(v):
    """ Normalize a vector to length 1. """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Wolf(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds three rules of flocking behavior;
    plus a fourth rule to avoid the edges of the simulation space. """

    def setup(self):

        self.velocity = normalize(
            self.model.random.random(self.p.dimensions) - 0.5)

    def update_velocity(self):

        pos = self.position()
        dim = self.p.dimensions

        # Rule 1 - Cohesion
        nbs = self.neighbors(distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_list = nbs.position()
        if nbs_len > 0:
            center = np.sum(nbs_pos_list, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(self.p.dimensions)

        # Rule 2 - Seperation
        v2 = np.zeros(dim)
        for nb in self.neighbors(distance=self.p.inner_radius):
            v2 -= nb.position() - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs.velocity, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(self.p.dimensions)

        # Rule 4 - Borders
        v4 = np.zeros(dim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(self.p.dimensions):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.env.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalize(self.velocity)

    def update_position(self):

        self.move_by(self.velocity)


class Wolfpack(ap.Model):
    """
    An agent-based model of animals' flocking behavior,
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].

    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    """

    def setup(self):
        """ Initializes the agents and network of the model. """

        self.space = self.add_space(shape=[self.p.size]*self.p.dimensions)
        self.space.add_agents(self.p.population, Wolf, random=True)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction



def animation_plot_single(m, ax):
    dim = m.p.dimensions
    ax.set_title(f"Boids Flocking Model {dim}D t={m.t}")
    pos = m.space.positions(transpose=True)
    ax.scatter(*pos, s=1, c='black')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if dim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()

def animation_plot(m, p):
    projection = '3d' if p['dimensions'] == 3 else None
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=projection)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    # return IPython.display.HTML(animation.to_jshtml(fps=20))
    return IPython.display.HTML(animation.to_jshtml(fps=20))

parameters2D = {
    'size': 50,
    'seed':123,
    'steps': 200,
    'dimensions': 2,
    'population': 200,
    'inner_radius': 3,
    'outer_radius': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'seperation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}

animation_plot(Wolfpack, parameters2D)