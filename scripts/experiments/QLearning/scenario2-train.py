from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment, ToyEnvironment1
import dill
from SSN_RL.agent.QAgent import QAgent
from SSN_RL.environment.rewards import reward_v1
import datetime
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
import numpy as np


start = datetime.datetime.now()
file_prefix = './scripts/experiments/QLearning/scenario2_v2'

EPISODES = 2500


env = Scenario2Environment()
agent = QAgent("agent1", env.satKeys, env.sensorKeys)


for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()

    state = agent.encodeState(t, stateCat)
    state_disc = agent.discretizeState(state)

    while not Done:
        # take actions
        action, actions_decoded, action_disc = agent.decide(t, events, stateCat)

        #print(state_disc)
        #print(action_disc)

        # get reward
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        # advance env
        t, events, stateCat, Done = env.step(actions_decoded)
        
        # next state
        next_state = agent.encodeState(t, stateCat)
        next_state_disc = agent.discretizeState(next_state)
        

        agent.updateQTable(state_disc, action_disc, reward, next_state_disc)
                

        state = next_state
        state_disc = next_state_disc

        total_reward += reward
    

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.eps_threshold:.4f}")


env.debug_ec.display()
print('Total time:',str((datetime.datetime.now() - start).total_seconds()/60), 'mins')
print(total_reward)


#print("NOT SAVING FILE")
with open(file_prefix+".pkl", 'wb') as f:
    dill.dump(agent.qTable, f)


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