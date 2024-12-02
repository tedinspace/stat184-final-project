import random
from skyfield.api import load

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.debug.Loggers import EventCounter

class Environment: 
    def __init__(self):
        print("TODO")
        # randomize epoch and scenario length
        self.sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0)+random.random(), random.uniform(12, 72))
        self.sConfigs.updateDT_careful(15) # change scenario time step

        self.satTruth = {} # TODO
        self.stateCatalog = StateCatalog(self.satTruth)
        self.t = self.sConfigs.scenarioEpoch

        self.debug_ec = EventCounter()
    

    def reset(self):
        # reset the environment/re-int a new run
        return # returns initial state

    def step(self, action, agent):
        return # next_state, reward, done, truncated, info
    
