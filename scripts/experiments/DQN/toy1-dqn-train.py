import torch
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.rewards import reward_v1
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1
from SSN_RL.agent.DQNAgent import DQNAgent

import numpy as np
import datetime
file_prefix = './scripts/experiments/DQN/dqn_toy1_v1'

env = ToyEnvironment1()
agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
num_episodes = 2

saved_rewards = []
saved_eps = []
start = datetime.datetime.now()
for episode in range(num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

    while not Done:
        # take actions
        action = agent.decide(state)

        action_spec =  np.round(action.numpy()).astype(int)
        #print(action_spec)
        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action_spec, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        agent.step(state, action, reward, next_state, Done)

        state = next_state
        total_reward += reward
    
    # decay epsilon 
    if (episode + 1) % 10 == 0:
        saved_rewards.append(total_reward)
        saved_eps.append(agent.eps_threshold)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.eps_threshold:.4f}")

end = datetime.datetime.now()
elapsed = end - start
print('Total time:',str(elapsed.total_seconds()/60), 'mins')

torch.save(agent.model.state_dict(), file_prefix+".pth")
with open(file_prefix+"_rewards.txt", 'w') as file:
    file.write(str([saved_rewards,saved_eps]))
