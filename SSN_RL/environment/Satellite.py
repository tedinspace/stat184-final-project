from skyfield.api import EarthSatellite
from utils.time import m2frac


class Satellite: 
    def __init__(self,name, l1, l2, sConfigs):
        # save original state
        self.l1 = l1
        self.l2 = l2

        # record of initial reepoching
        self.reEpoched = False # has the TLE epoch been overriden? 
        self.nEpoch = []
        
        # usable objects
        self.activeObject = EarthSatellite(l1, l2, name, sConfigs.ts)

        # for referencing
        self.timeScale =  sConfigs.ts
        self.scenarioEpoch =  sConfigs.scenarioEpoch

    def reEpoch_init(self, timeLastSeenMinsFromScenarioEpoch):
        '''re epoching allows us to change the state's epoch time without changing the TLE string
        - this gives us better control over our satellites in the scenario'''
        self.reEpoched = True
        self.nEpoch = self.scenarioEpoch - m2frac(timeLastSeenMinsFromScenarioEpoch)
        self.activeObject.epoch = self.nEpoch
    
    def tick(self, t):
        return "TODO"
