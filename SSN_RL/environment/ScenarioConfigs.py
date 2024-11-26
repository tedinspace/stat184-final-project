from skyfield.api import load
from SSN_RL.utils.time import SPD, h2frac
class ScenarioConfigs:
    def __init__(self, scenarioEpoch, scenarioLengthHours):
        
        
        self.ts  = load.timescale()

        self.dt = 5 # [s] - time step
        self.scenarioEpoch = scenarioEpoch
        self.scenarioLengthHours = scenarioLengthHours
        self.scenarioEnd = self.scenarioEpoch + h2frac(self.scenarioLengthHours)

        self.timeDelta = self.dt / SPD
        # number of total iterations
        self.nSteps = round(self.scenarioLengthHours*60*(60/self.dt))

    def updateDT_careful(self, dt):
        '''change the delta t; shouldn't do this in the middle of a scenario'''
        self.dt = dt
        self.timeDelta = self.dt / SPD
        self.nSteps = round(self.scenarioLengthHours*60*(60/self.dt))

