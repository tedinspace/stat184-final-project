from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1, ToyEnvironment1_generalization_test_1
from SSN_RL.agent.HeuristicAgent import HeuristicAgent
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
from SSN_RL.environment.rewards import reward_v1

import datetime


start = datetime.datetime.now()
env = ToyEnvironment1()

t, events, stateCat, Done = env.reset(deltaT=60*5)


agent = HeuristicAgent("agent1", env.satKeys,env.sensorKeys)
total_reward = 0
while not Done:
    # take actions
    action = agent.decide(t, events, stateCat)
    t, events, stateCat, Done = env.step({agent.agentID: action})
    reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
    total_reward += reward
    
elapsed = datetime.datetime.now() - start
print('Total time:',str(elapsed.total_seconds()), ' [s]')

# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -
env.debug_ec.display()
nManeuvers = 0
for satKey in env.satTruth:
    nManeuvers+= env.satTruth[satKey].nManeuvers

print("actual unique maneuvers: "+str(nManeuvers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))
print("REWARD: "+str(total_reward))

fig, ax = plt.subplots()
colors = {
    'MUOS1': 'blue', 
    'MUOS2': 'red',
    "MUOS3": 'orange', 
    "MUOS4": 'green',
    "MUOS5": 'purple', 
    'AEHF 1 (USA 214)': 'red', 
    'AEHF 2 (USA 235)': 'blue'

}

sEpoch = env.sConfigs.scenarioEpoch

i = 0
for sensor in env.sensorMap:
    for task in env.sensorMap[sensor].completedTasks:
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
