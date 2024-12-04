import matplotlib.pyplot as plt

from SSN_RL.environment.Agent import AgentWrapper
from SSN_RL.environment.Environment import Environment
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR, MAUI
from SSN_RL.utils.time import hrsAfterEpoch
from SSN_RL.utils.vis import seenAndUnseenAtSensors
from SSN_RL.utils.struct import getNames


sensorList = [MHR, MAUI]
# make sure all satellites can be seen 
seen, _, stateList = seenAndUnseenAtSensors(MUOS_CLUSTER, sensorList)
A = [AgentWrapper("agent 1", seen, getNames(sensorList))]

env = Environment(stateList, sensorList)
t, events, stateCat, Done = env.reset()

while Done ==False:

    actions = {}
    for agent in A:
        actions[agent.agentID]=agent.decide(t, events, stateCat)
    t, events, stateCat, Done = env.step(actions)



# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -

env.debug_ec.display()
nManuevers = 0
for satKey in env.satTruth:
    nManuevers+= len(env.satTruth[satKey].maneuverList)
print("actual unique maneuvers: "+str(nManuevers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))

fig, ax = plt.subplots()
colors = {
    'MUOS1': 'blue', 
    'MUOS2': 'red',
    "MUOS3": 'orange', 
    "MUOS4": 'green',
    "MUOS5": 'purple'
}

sEpoch = env.sConfigs.scenarioEpoch

i = 0
for sensor in sensorList:
    for task in env.sensorMap[sensor.name].completedTasks:
        ax.plot([hrsAfterEpoch(sEpoch, task.startTime),hrsAfterEpoch(sEpoch,task.stopTime) ], [i, i], color=colors[task.satID], linewidth=4)
    i+=1
for satKey in env.satTruth:
    for m in env.satTruth[satKey].maneuverList:
        ax.axvline(x=hrsAfterEpoch(sEpoch,m.time), color=colors[satKey], linestyle=':', linewidth=2)
for event in env.debug_uniqueManeuverDetections:
    ax.axvline(x=hrsAfterEpoch(sEpoch,event.arrivalTime), color=colors[event.satID], linestyle='-.', linewidth=2)

plt.title('Plotted Results of Randomized Actions')
plt.ylabel('Executed Schedule at Sensor (MHR)')
plt.xlabel('time [hours after scenario epoch]')
plt.show()