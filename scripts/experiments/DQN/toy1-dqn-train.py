import torch
from SSN_RL.agent.functions.encode import encode_basic_v2
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.utils.time import MPD
import numpy as np
import datetime

def reward_custom(t, events, stateCatalog, agentID, sat2idx, last_tasked):
    rewardOrCost = 0
    last_tasked = (t.tt - last_tasked)*MPD
    rewardOrCost -= np.sum(last_tasked < 15) * 2
    
    for event in events:
        if event.agentID == agentID:
            
            if event.type == SensorResponse.INVALID:
                rewardOrCost -= 50
            if event.type == SensorResponse.INVALID_TIME:
                rewardOrCost -= 2
            elif event.type == SensorResponse.DROPPED_SCHEDULING:
                rewardOrCost -= 2
            elif event.type == SensorResponse.DROPPED_LOST:
                rewardOrCost -= 1000
            elif event.type == SensorResponse.COMPLETED_NOMINAL:
                rewardOrCost += 1
            elif event.type == SensorResponse.COMPLETED_MANEUVER:
                rewardOrCost += 4
    
    lastSeen = np.array([
        stateCatalog.lastSeen_mins(t, sat) 
        for sat in sat2idx.keys()
    ])
    
    rewardOrCost -= np.sum(lastSeen > 90) * 2
    
    return rewardOrCost

file_prefix = './scripts/experiments/DQN/dqn_toy1_v1'

env = ToyEnvironment1()
agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
num_episodes = 100

saved_rewards = []
saved_eps = []
start = datetime.datetime.now()

last_tasked = np.ones(env.nSats)
for episode in range(num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v2(t, events, stateCat, agent.agentID, agent.sat2idx)

    while not Done:
        # take actions
        action = agent.decide(state)

        action_spec =  np.round(action.numpy()).astype(int)
        mask = action_spec != -1

        # Update arr2 at the positions where arr1 is not -1
        last_tasked[mask] = np.ones(env.nSats)[mask]*t.tt

        #print(action_spec)
        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action_spec, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v2(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_custom(t, events, stateCat, agent.agentID, agent.sat2idx, last_tasked)
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


