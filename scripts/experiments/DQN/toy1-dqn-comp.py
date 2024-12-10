
import torch
from SSN_RL.agent.algorithms.trivial import randomAction,noAction
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1,ToyEnvironment1_generalization_test_1
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.rewards import reward_v1
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch

import numpy as np

n_runs = 1000
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


DQN_rewards = []


sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}
for episode in range(n_runs):
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
    DQN_rewards.append(total_reward)
    
     
DO_NOTHING_rewards =[]
for episode in range(n_runs):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    while not Done:
        action = noAction(nSats)
        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        state = next_state
        total_reward += reward
    DO_NOTHING_rewards.append(total_reward)

RANDOM_rewards =[]
for episode in range(n_runs):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    while not Done:
        action = randomAction(nSensors, nSats)
        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        state = next_state
        total_reward += reward
    RANDOM_rewards.append(total_reward)


#print(RANDOM_rewards)
#print(DO_NOTHING_rewards)
#print(DQN_rewards)   

data = [RANDOM_rewards, DO_NOTHING_rewards, DQN_rewards]

# Create a single box plot
colors = ['#69b3a2', '#ff9999', '#6699cc']

# Create the box plot with customized colors
boxplot = plt.boxplot(data, patch_artist=True, 
            boxprops=dict(facecolor=colors[0], color='black'), 
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='red', marker='o'),
            medianprops=dict(color='black'))

for i in range(len(boxplot['boxes'])):
    boxplot['boxes'][i].set_facecolor(colors[i])

# Set titles and labels
plt.title('Side-by-Side Box Plots', fontsize=16, fontweight='bold')
plt.xlabel('Datasets', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Customize the x-axis labels
plt.xticks([1, 2, 3], ['Random Policy', 'Do Nothing Policy', 'DQN'])

# Show the plot
plt.tight_layout()
plt.show()
