from skyfield.api import load
import matplotlib.pyplot as plt

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.Satellite import Satellite, Maneuver
from SSN_RL.environment.Agent import AgentWrapper
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.environment.Environment import Environment
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR, MAUI
from SSN_RL.debug.Loggers import EventCounter
from SSN_RL.utils.time import hrsAfterEpoch, defaultEpoch
from SSN_RL.utils.vis import seenAndUnseenAtSensors


sensorList = [MHR, MAUI]
# make sure all satellites can be seen 
seen, _, stateList = seenAndUnseenAtSensors(MUOS_CLUSTER, sensorList)
print(seen)

env = Environment(stateList, sensorList)