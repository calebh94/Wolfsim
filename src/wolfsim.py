'''
WolfSim - The Wolf Tracking Simulator #1

by Caleb Harris and Johnie Sublett
GT M&S Spring 2021

'''

import simpy


def wolf(env):
    while True:
        print("Start eating at %d" % env.now)
        eating_duration = 5
        yield env.timeout(eating_duration)

        print("Start roaming at %d" % env.now)
        roaming_duration = 20
        yield env.timeout(roaming_duration)

env = simpy.Environment()
env.process(wolf(env))
env.run(until=500)



