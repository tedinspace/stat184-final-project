import torch
from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.environment.rewards import reward_v1
import datetime
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
import numpy as np


start = datetime.datetime.now()
file_prefix = './scripts/experiments/DQN/dqn_scenario2_1'

EPISODES = 1000


env = Scenario2Environment()
agent = DQNAgent("agent1", env.satKeys,env.sensorKeys,  epsilon=1, epsilon_min=.3)

start = datetime.datetime.now()

for episode in range(EPISODES):
    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    state = agent.encodeState(t, stateCat)

    while not Done:
        # take actions
        action, actions_decoded = agent.decide(t, events, stateCat)

        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        t, events, stateCat, Done = env.step(actions_decoded)
        next_state = agent.encodeState(t, stateCat)
                
        agent.step(state, action, reward, next_state, Done)

        state = next_state
        total_reward += reward
    
    if episode % 2 == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())
    if (episode + 1) % 10 == 0:

        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.eps_threshold:.4f}")


env.debug_ec.display()
print('Total time:',str((datetime.datetime.now() - start).total_seconds()/60), 'mins')
print(total_reward)


torch.save(agent.model.state_dict(), file_prefix+".pth")



env.debug_ec.display()
nManuevers = 0
for satKey in env.satTruth:
    nManuevers+= len(env.satTruth[satKey].maneuverList)
print("actual unique maneuvers: "+str(nManuevers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))



fig, ax = plt.subplots()


def string_to_color(s):

    np.random.seed(len(s))  # Set the seed to ensure consistency for the same string
    
    return np.random.rand(3,) 
c = ["black", "#29A634", "#D1980B", "#D33D17", "#9D3F9D", "#00A396", "#DB2C6F", "#8EB125", "#946638", "#7961DB"]
colors = {}
for i in range(len(env.satKeys)):
    colors[env.satKeys[i]]=c[i]






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


plt.ylabel('Executed Schedule at Sensor (MHR)')
plt.xlabel('time [hours after scenario epoch]')
plt.show()