
from SSN_RL.environment.Environment import Environment
from SSN_RL.scenarioBuilder.Randomizer import Randomizer
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR

def ToyEnvironment1():
    # 6 hours; 1 RADAR; 2 objects; each maneuvers once
    states = [MUOS_CLUSTER[0], MUOS_CLUSTER[2]]
    sensor = [MHR]
    R = Randomizer()
    R.maneuverProb=1
    R.maneuverCountRange = [1,1]
    R.scenarioLengthRange = [6,6]
    return Environment(states, sensor, R)

