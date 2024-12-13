import torch
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.environment.rewards import reward_v1
import datetime


file_prefix = './scripts/experiments/DQN/dqn_toy1_v3'
#print("FILE SAVE IS COMMENTED OUT--- WARNING")
env = ToyEnvironment1()

agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
num_episodes = 2500

saved_rewards = []
saved_eps = []
start = datetime.datetime.now()

for episode in range(num_episodes):
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
        saved_rewards.append(total_reward)
        saved_eps.append(agent.eps_threshold)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.eps_threshold:.4f}")

    #elapsed = datetime.datetime.now() - start
    #print('Total time:',str((datetime.datetime.now() - start).total_seconds()), ' [s]')

end = datetime.datetime.now()
elapsed = end - start
print('Total time:',str(elapsed.total_seconds()/60), 'mins')
print(total_reward)

torch.save(agent.model.state_dict(), file_prefix+".pth")
with open(file_prefix+"_rewards.txt", 'w') as file:
    file.write(str([saved_rewards,saved_eps]))


