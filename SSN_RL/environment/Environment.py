import random
from skyfield.api import load

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.Randomizer import Randomizer
from SSN_RL.debug.Loggers import EventCounter

class Environment: 
    def __init__(self, inputTleList, sensorList):
        self.R = Randomizer()
        # randomize epoch and scenario length
        self.sConfigs = self.R.randomizeScenarioSpecs().updateDT_careful(15)

        self.satTruth = self.R.randomizeSatTruth(inputTleList, self.sConfigs)

        self.stateCatalog = StateCatalog(self.satTruth)
        
        self.t = self.sConfigs.scenarioEpoch

        # save
        self.inputTleList = inputTleList
        # debug
        self.debug_ec = EventCounter()
    

    def reset(self):
        # reset the environment/re-int a new run
        return # returns initial state

    def step(self, action, agent):
        return # next_state, reward, done, truncated, info
    
