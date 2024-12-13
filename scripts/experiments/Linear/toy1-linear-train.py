from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1
from SSN_RL.agent.LinearAgent import LinearQAgent
from SSN_RL.environment.rewards import reward_v1
import numpy as np
import datetime
import dill
import matplotlib.pyplot as plt

EPISODES = 1000
file_prefix = './scripts/experiments/Linear/linear_toy1_v1'
start = datetime.datetime.now()

env = ToyEnvironment1()
agent = LinearQAgent(
    agentID="agent1",
    assigned_sats=env.satKeys,
    assigned_sensors=env.sensorKeys,
    learning_rate=0.01,
    gamma=0.99,
    epsilon=1.0,
    epsilon_dec=0.999,
    epsilon_min=0.05
)

reward_history = []
episode_lengths = []

for episode in range(EPISODES):
    total_reward = 0
    steps = 0
    
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    
    state = agent.encodeState(t, stateCat)
    
    while not Done:
        actions, actions_decoded = agent.decide(t, events, stateCat)
        
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        
        t, events, stateCat, Done = env.step(actions_decoded)
        steps += 1
    
    reward_history.append(total_reward)
    episode_lengths.append(steps)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_history[-10:])
        avg_length = np.mean(episode_lengths[-10:])
        print(f"Episode {episode + 1}/{EPISODES}")
        print(f"Average Reward (last 10): {avg_reward:.2f}")
        print(f"Average Episode Length (last 10): {avg_length:.2f}")
        print(f"Epsilon: {agent.eps_threshold:.4f}")
        print("----------------------------------------")

print(f'Total training time: {(datetime.datetime.now() - start).total_seconds()/60:.2f} mins')
print(f'Final average reward (last 100): {np.mean(reward_history[-100:]):.2f}')

save_data = {
    'weights': agent.weights,
    'reward_history': reward_history,
    'episode_lengths': episode_lengths,
    'training_time': (datetime.datetime.now() - start).total_seconds(),
    'hyperparameters': {
        'learning_rate': agent.alpha,
        'gamma': agent.gamma,
        'initial_epsilon': 1.0,
        'epsilon_decay': agent.epsilon_dec,
        'epsilon_min': agent.epsilon_min
    }
}

print(f'Saving model to {file_prefix}.pkl')
with open(file_prefix + '.pkl', 'wb') as f:
    dill.dump(save_data, f)


plt.figure(figsize=(10, 5))
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'))
plt.title('Rolling Average Reward (window=100)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig(file_prefix + '_rewards.png')
plt.close()

# Plot episode lengths
plt.figure(figsize=(10, 5))
plt.plot(np.convolve(episode_lengths, np.ones(100)/100, mode='valid'))
plt.title('Rolling Average Episode Length (window=100)')
plt.xlabel('Episode')
plt.ylabel('Steps per Episode')
plt.savefig(file_prefix + '_lengths.png')
plt.close()

