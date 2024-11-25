from skyfield.api import load
from SSN_RL.utils.time import SPD, h2frac
class ScenarioConfigs:
    def __init__(self, scenarioEpoch, scenarioLengthHours):
        
        #self.taskDelay = 10 # [mins] - time for message to be received by sensor from agent 
        #self.taskDelayRand = [-5,5] # [-mins, +mins] randomness added tasking delay 
        self.ts  = load.timescale()

        self.dt = 5 # [s] - time step
        self.scenarioEpoch = scenarioEpoch
        self.scenarioLengthHours = scenarioLengthHours
        self.scenarioEnd = self.scenarioEpoch + h2frac(self.scenarioLengthHours)

        self.timeDelta = self.dt / SPD
        # number of total iterations
        self.nSteps = round(self.scenarioLengthHours*60*(60/self.dt))



        
