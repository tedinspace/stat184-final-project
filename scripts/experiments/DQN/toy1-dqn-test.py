
import torch
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1,ToyEnvironment1_generalization_test_1
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.rewards import reward_v1
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch

import numpy as np

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

for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    while not Done:
        #action = randomAction(nSensors, nSats)
        action = agent.decide_trained(state)
        action_spec =  np.round(action.numpy()).astype(int)

        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action_spec, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        state = next_state
        total_reward += reward
    
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
