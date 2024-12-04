import random

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Satellite import Satellite, Maneuver

from SSN_RL.utils.time import defaultEpoch

class Randomizer:
    def __init__(self):
        self.defaultEpoch = defaultEpoch

        self.maneuverProb = 1/3 
        self.maneuverCountRange = [1,3] # [number of maneuvers]
        self.maneuverMagnitudeRange = [0.5, 25] # []
        self.scenarioLengthRange = [12,48] # [hours]
        self.reEpochRange = [30, 12*60] # [mins]
        
        
        
    def randomizeEpoch(self):
        '''add some random time to the default epoch'''
        return self.defaultEpoch + random.random()
    
    def randomizeScenarioLength(self):
        '''randomize scenario length; returns length in hours'''
        return random.uniform(self.scenarioLengthRange[0],self.scenarioLengthRange[1])

    def randomizeScenarioSpecs(self):
        ''''''
        return ScenarioConfigs(self.randomizeEpoch(), self.randomizeScenarioLength())

    def randomizeReEpoching(self, inputTleList, sConfigs):
        ''''''
        S = {}
        for tle in inputTleList:
            sat =  Satellite(tle[0], tle[1], tle[2], sConfigs )
            sat.reEpoch_init(random.uniform(self.reEpochRange[0],self.reEpochRange[1]))
            S[sat.name] = sat
        return S
    def randomizeManeuvers(self, satTruth, sConfigs):
        ''''''
        for sKey in satTruth:
            if random.random() < self.maneuverProb:
                nManeuvers = random.randint(self.maneuverCountRange[0], self.maneuverCountRange[1])
                for _ in range(nManeuvers):
                    mTime = random.uniform(0, sConfigs.scenarioLengthHours -1 )
                    mag = random.uniform(self.maneuverMagnitudeRange[0], self.maneuverMagnitudeRange[1])
                    satTruth[sKey].addManeuvers([Maneuver(mag,mTime, sConfigs)])
                    
        return satTruth
    
    def randomizeSatTruth(self, inputTleList, sConfigs):
        ''''''
        return self.randomizeManeuvers(self.randomizeReEpoching(inputTleList, sConfigs), sConfigs)
    




