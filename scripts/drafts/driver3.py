import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.algorithms.NN import QNetwork_Shallow
from SSN_RL.agent.algorithms.trivial import randomAction
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.Environment import Environment
from SSN_RL.environment.rewards import reward_v1
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR, MAUI, ASCENSION
from SSN_RL.utils.vis import seenAndUnseenAtSensors
from SSN_RL.utils.struct import getNames


def update_q_network(state, action, reward, next_state, done, q_network, optimizer, gamma=0.99):
    # Get Q-values for the current state (shape: [batch_size, action_space_size])
    #print("q_values", q_values)
    # Convert action to indices based on min_action (shape: [batch_size])
    action_indices = [int(a) - q_network.min_action for a in action]  # Convert actions to indices
    q_values = q_network(state)

    # Get the target Q-value from the next state
    with torch.no_grad():
        next_q_values = q_network(next_state)  # Get Q-values for the next state
        next_q_max = torch.max(next_q_values)  # Get the maximum Q-value for the next state

    # Create a target tensor (same shape as q_values) and update the Q-value for the action taken
    target = q_values.clone()  # Clone the Q-values to modify the action taken
    target[0, action_indices] = reward + (1 - done) * gamma * next_q_max  # Update the Q-value for the action taken

    # Calculate loss (mean squared error)
    loss = nn.MSELoss()(q_values[0, action_indices], target[0, action_indices])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# create environment
sensorList = [MHR, MAUI, ASCENSION]
sensorKeys = getNames(sensorList)
satKeys, _, stateList = seenAndUnseenAtSensors(MUOS_CLUSTER, sensorList)
env = Environment(stateList, sensorList)


nSensors = len(sensorList)
nSats = len(satKeys)
input_dim = nSats*2
output_dim = nSats

q_network = QNetwork_Shallow(input_dim, output_dim, [-1, nSensors-1])
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
# training
TS = TrainingSpecs()
TS.epsilon_decay = 0.9995
epsilon = .1

agentID = "agent1"
sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}

for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    while not Done:
        # take actions
        if random.uniform(0,1) < epsilon:
             action = randomAction(nSensors, nSats)
        else:
             with torch.no_grad():
                 q_values = q_network(state)
                 action =  np.round(q_values.numpy()).astype(int)[0]


        t, events, stateCat, Done = env.step({agentID: decodeActions(action, satKeys, sensorKeys)})
        next_state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agentID, sat2idx)
        update_q_network(state, action, reward, next_state, Done, q_network, optimizer)

        state = next_state
        total_reward += reward
    
    # decay epsilon 
    epsilon = max(TS.min_epsilon, epsilon * TS.epsilon_decay)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        

torch.save(q_network.state_dict(), "q_network.pth")

