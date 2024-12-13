
from SSN_RL.environment.Environment import Environment
from SSN_RL.scenarioBuilder.Randomizer import Randomizer
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER, AEHF_CLUSTER, INTELSAT_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR, ASCENSION, SOCORRO
from SSN_RL.utils.vis import seenAndUnseenAtSensors

def ToyEnvironment1():
    # 6 hours; 1 RADAR; 2 objects; each maneuvers once
    states = [MUOS_CLUSTER[0], MUOS_CLUSTER[2]]
    sensor = [MHR]
    R = Randomizer()
    R.maneuverProb=1
    R.maneuverCountRange = [1,1]
    R.scenarioLengthRange = [6,6]
    return Environment(states, sensor, R) # set timestep on .reset(deltaT=value)


def ToyEnvironment1_generalization_test_1():
    # see how agents trained on toy 1 generalize in a more complicated environment with different objects and sensor
    states = [AEHF_CLUSTER[0], AEHF_CLUSTER[1]]
    sensor = [ASCENSION]
    R = Randomizer()
    R.maneuverProb=1
    R.maneuverCountRange = [2,4]
    R.scenarioLengthRange = [48,48]

    return Environment(states, sensor, R)


def Scenario2Environment():
    sensor = [ASCENSION, MHR, SOCORRO]
    _, _, states = seenAndUnseenAtSensors(MUOS_CLUSTER+AEHF_CLUSTER, sensor)
    R = Randomizer()
    R.maneuverProb=1
    R.maneuverCountRange = [1,4]
    R.scenarioLengthRange = [6,12]
    return Environment(states, sensor, R)

def Scenario2Environment_generalization_test():
    sensor = [ASCENSION, MHR, SOCORRO]
    _, _, states = seenAndUnseenAtSensors(MUOS_CLUSTER+AEHF_CLUSTER, sensor)
    R = Randomizer()
    R.maneuverProb=1
    R.maneuverCountRange = [0,5]
    R.scenarioLengthRange = [32,48]
    return Environment(states, sensor, R)