'''
WolfSim - The Wolf Tracking Simulator #1

by Caleb Harris and Johnie Sublett
GT M&S Spring 2021

'''

import simpy
#TODO: test with agentpy as ap


class Wolf(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
        self.alive = True
        self.position = [0,0]
        self.speed = 5

    def run(self):
        while self.alive:
            print("Start eating at %d" % self.env.now)
            eating_duration = 5
            # yield env.timeout(eating_duration)\
            try:
                yield self.env.process(self.eat(eating_duration))
            except simpy.Interrupt:
                print("Sim interrupted. Wolf tracking may have failed or wolf has become deceased")
                self.alive = False

            print("Start roaming at %d" % env.now)
            roaming_duration = 20
            # yield env.timeout(roaming_duration)
            yield self.env.timeout(roaming_duration)

    def eat(self, duration):
        yield self.env.timeout(duration)


def bear(env, wolf):
    yield env.timeout(30)
    wolf.action.interrupt()


env = simpy.Environment()
wolf = Wolf(env)
env.process(bear(env, wolf))
env.run(until=200)




