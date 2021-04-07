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
        if self.pos is None:
            return
        step_options = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)
        new_position = self.move_decision(step_options)
        # new_position = self.random.choice(step_options)
        self.model.grid.move_agent(self, new_position)
        #TODO: ADD WEIGHTS BASED ON TERRAIN AND VEGETATION FOR MOVEMENT

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
            self.pos = None
        if not self.alive:
            return
        if self.age > 8:
            self.attack()
        if debug:
            print("Agent " + str(self.unique_id) + " with age " + str(self.age) +" is moving from position " + str(self.pos))
        self.age += 1
        self.move()

    def move_decision(self, options):
        # current_position = self.neighbor_choice(options, masterdict)
        new_position = self.random.choice(options)
        return new_position

    def check_vegetation_of_neighbor(self, neighborlist, neighbordict):
        # returns a list of neighbors as vegetation types
        neighbor_veg = {}
        neighbor_veg_list = []
        for neighbor in neighborlist:
            for nposlist in neighbordict.values():  # from all the grid values, find neighbors for this particular grid
                for neighbor_position in nposlist:  # in order to find out neighbor's vegetation -> neighbor's weighted
                    if neighbor == neighbor_position:  # value -> selected neighbor to move to.
                        vegetation = list(neighbordict.keys())[list(neighbordict.values()).index(nposlist)]
                        neighbor_veg.setdefault(neighbor, []).append(vegetation)
        for list_of_values in neighbor_veg.values():
            if len(list_of_values) > 1:  # if there is more than one land type at that grid,
                for value in list_of_values:
                    if value != 'Elevation_Out_of_Bound' and value != 'Outside_FNNR' \
                            and value != 'PES' and value != 'Forest' and value != 'Farm'\
                            and value != 'Household':
                        if len(list_of_values) > 1:  # checking again in case this loops multiple times
                            list_of_values.remove(value)
                        # Vegetation is considered the bottom layer, so in case of a conflict, it is removed.
                    elif value == 'Outside_FNNR':
                        list_of_values = ['Outside_FNNR']  # Otherwise, Outside_FNNR is the defining layer;
                    elif value == 'Elevation_Out_of_Bound':
                        list_of_values = ['Elevation_Out_of_Bound']  # then the other layers follow in order of
                    elif value == 'Household':  # formation.
                        list_of_values = ['Household']
                    elif value == 'Farm':
                        list_of_values = ['Farm']
                    elif value == 'PES':
                        list_of_values = ['PES']
                    elif value == 'Forest':
                        list_of_values = ['Forest']
            for value in list_of_values:
                neighbor_veg_list.append(value)
        return neighbor_veg_list

    def neighbor_choice(self, neighborlist, neighbordict):
        # agent chooses a neighbor to move to based on weights
        choicelist = []
        # picks a weighted neighbor to move to
        # neighbordict is a dictionary with all vegetation categories and their corresponding grid values
        # neighborlist is a list of 8-cell neighbors to the current position
        neighbor_veg = self.check_vegetation_of_neighbor(neighborlist, neighbordict)
        # weights below were taken from the pseudocode, and can be modified
        # for better code, a dictionary can be created instead of the below structure
        for vegetation in neighbor_veg:
            if vegetation == 'Elevation_Out_of_Bound':
                weight = 0
            elif vegetation == 'Bamboo':
                weight = 0.8
            elif vegetation == 'Coniferous':
                weight = 1
            elif vegetation == 'Broadleaf':
                weight = 1
            elif vegetation == 'Mixed':
                weight = 1
            elif vegetation == 'Lichen':
                weight = 0.8
            elif vegetation == 'Deciduous':
                weight = 1
            elif vegetation == 'Shrublands':
                weight = 0.8
            elif vegetation == 'Clouds':
                weight = random.uniform(0, 1)
            elif vegetation == 'Farmland':
                weight = 0
            elif vegetation == 'Outside_FNNR':
                weight = 0
            elif vegetation == 'Household':
                weight = 0
            elif vegetation == 'Farm':
                weight = 0.05
            elif vegetation == 'PES':
                weight = 0.2
            elif vegetation == 'Forest':
                weight = 0.3
            choicelist.append(weight)

        if choicelist != [] and choicelist != [0, 0, 0, 0, 0, 0, 0, 0]:
            # this takes care of edges
            while len(choicelist) < 8:
                choicelist.append(0)
            # random choice plays a role, but each neighbor choice is affected by weights
            # the next few dozen lines determine which weighted % category the random choice falls into

            # For example: if the relative weight of the north neighbor was 0.16 (with the sum of all weights
            # equaling 1) and the relative weight of the south neighbor was 0.08, then oldsum < chance < newsum
            # for the north neighbor would be 0 < x (random decimal from 0 to 1) < 0.16, and oldsum < chance < newsum
            # for the south neighbor follows that, from 0.16 < x < 0.24. The exact order does not matter; for the
            # north neighbor, whose relative weight is 0.16, we could also write something like 0.08 < x < 0.24
            # after assessing the weight of the south neighbor first (0 < x < 0.08), so long as the range (difference
            # between high and low values) for the north neighbor was 0.16, etc.
            #  and all of the continuous ranges summed up to 16.

            # a new weighted-choice function in Python 3.6 eliminates the need for this formula; however,
            # this project will not change its dependencies at this point (Python 3.6 is required for this choice
            # function).
            chance = random.uniform(0, 1)

            oldsum = 0
            newsum = choicelist[1] / sum(choicelist)  # defines the relative weight of the north neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[1]  # the north neighbor is selected
            oldsum = newsum
            newsum += choicelist[6] / sum(choicelist)  # defines the relative weight of the south neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[6]  # the south neighbor is selected
            oldsum = newsum
            newsum += choicelist[3] / sum(choicelist)  # defines the relative weight of the west neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[3]  # the west neighbor is selected
            oldsum = newsum
            newsum += choicelist[4] / sum(choicelist)  # defines the relative weight of the east neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[4]  # the east neighbor is selected
            oldsum = newsum
            newsum += choicelist[0] / sum(choicelist)  # defines the relative weight of the northwest neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[0]  # the northwest neighbor is selected
            oldsum = newsum
            newsum += choicelist[2] / sum(choicelist)  # defines the relative weight of the northeast neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[2]  # the northeast neighbor is selected
            oldsum = newsum
            newsum += choicelist[5] / sum(choicelist)  # defines the relative weight of the southwest neighbor
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[5]  # the southwest neighbor is selected
            oldsum = newsum
            newsum += choicelist[7] / sum(choicelist)  # defines the relative weight of the southeast neighbor
            if newsum != 1 and newsum > 0.999:
                newsum = 1
            if oldsum < chance < newsum:  # if the randomly-chosen number falls into this weight's range,
                direction = neighborlist[7]  # the southeast neighbor is selected
            try:
                assert int(newsum) == 1
            except AssertionError:
                if self.current_position is not None:
                    direction = self.current_position
                else:
                    direction = self.saved_position
            return direction