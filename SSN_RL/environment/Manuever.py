from SSN_RL.utils.time import h2frac
import numpy as np
class Maneuver:
    def __init__(self,magDV, hoursIntoScenario, sConfigs):
        self.dir = np.array([.1, .8, .1]) # sums to 1
        self.maneuver = self.dir * magDV
        self.time = sConfigs.scenarioEpoch + h2frac(hoursIntoScenario)
        self.occurred = False

