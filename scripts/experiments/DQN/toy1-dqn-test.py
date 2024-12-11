
import torch
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1,ToyEnvironment1_generalization_test_1
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.environment.rewards import reward_v1
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
import datetime

env = ToyEnvironment1()

satKeys = env.satKeys
sensorKeys = env.sensorKeys

nSensors = env.nSensors
nSats = env.nSats
input_dim = nSats*2
output_dim = nSats


agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
agent.model.load_state_dict(torch.load("./scripts/experiments/DQN/dqn_toy1_v1.pth", weights_only=True))



agentID = "agent1"
sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}


TS = TrainingSpecs()
TS.num_episodes=1
epsilon = .1
start = datetime.datetime.now()
for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=60*5)
    state = agent.encodeState(t, stateCat)
    i = 0
    while not Done:
        action, actions_decoded = agent.decide_testing(t, events, stateCat)
        

        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        t, events, stateCat, Done = env.step(actions_decoded)

        next_state = agent.encodeState(t, stateCat)
        
        start = datetime.datetime.now()
        agent.step(state, action, reward, next_state, Done)
        #print('Total time:',str((datetime.datetime.now() - start).total_seconds()), ' [s]')

        state = next_state
        total_reward += reward
        i+=1
    
    print(i)
    # decay epsilon 
    epsilon = max(TS.min_epsilon, epsilon * TS.epsilon_decay)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        

# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -

env.debug_ec.display()
nManuevers = 0
for satKey in env.satTruth:
    nManuevers+= len(env.satTruth[satKey].maneuverList)
print("actual unique maneuvers: "+str(nManuevers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))
print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")   

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
