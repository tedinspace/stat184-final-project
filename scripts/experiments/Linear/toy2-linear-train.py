from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment
from SSN_RL.agent.LinearAgent import LinearQAgent
from SSN_RL.environment.rewards import reward_v1
import numpy as np
import datetime
import dill
import os
import matplotlib.pyplot as plt

# Configuration
EPISODES = 1000
file_prefix = './scripts/experiments/Linear/toy2_linear_v1'
start = datetime.datetime.now()

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(file_prefix), exist_ok=True)

# Initialize environment and agent
env = Scenario2Environment()  # Using scenario 2
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

# Training metrics
reward_history = []
episode_lengths = []
lost_satellites_history = []

print(f"Training on Scenario 2 with {len(env.satKeys)} satellites and {len(env.sensorKeys)} sensors")

for episode in range(EPISODES):
    total_reward = 0
    steps = 0
    lost_satellites = 0
    
    # Reset environment and agent
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    
    # Initial state
    state = agent.encodeState(t, stateCat)
    
    while not Done:
        # Take action
        actions, actions_decoded = agent.decide(t, events, stateCat)
        
        # Get reward and advance environment
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        t, events, stateCat, Done = env.step(actions_decoded)
        
        # Update metrics
        total_reward += reward
        steps += 1
        
        # Count lost satellites from events
        for event in events:
            if event.agentID == agent.agentID and hasattr(event, 'type'):
                if event.type == 'DROPPED_LOST':
                    lost_satellites += 1
    
    # Record metrics
    reward_history.append(total_reward)
    episode_lengths.append(steps)
    lost_satellites_history.append(lost_satellites)
    
    # Progress logging
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_history[-10:])
        avg_length = np.mean(episode_lengths[-10:])
        avg_lost = np.mean(lost_satellites_history[-10:])
        print(f"\nEpisode {episode + 1}/{EPISODES}")
        print(f"Average Reward (last 10): {avg_reward:.2f}")
        print(f"Average Episode Length (last 10): {avg_length:.2f}")
        print(f"Average Lost Satellites (last 10): {avg_lost:.2f}")
        print(f"Epsilon: {agent.eps_threshold:.4f}")
        print("----------------------------------------")

# Training summary
training_time = (datetime.datetime.now() - start).total_seconds()/60
print(f'\nTraining Complete!')
print(f'Total training time: {training_time:.2f} mins')
print(f'Final average reward (last 100): {np.mean(reward_history[-100:]):.2f}')
print(f'Final average lost satellites (last 100): {np.mean(lost_satellites_history[-100:]):.2f}')

# Save trained weights and training history
save_data = {
    'weights': agent.weights,
    'reward_history': reward_history,
    'episode_lengths': episode_lengths,
    'lost_satellites_history': lost_satellites_history,
    'training_time': training_time,
    'hyperparameters': {
        'learning_rate': agent.alpha,
        'gamma': agent.gamma,
        'initial_epsilon': 1.0,
        'epsilon_decay': agent.epsilon_dec,
        'epsilon_min': agent.epsilon_min
    },
    'env_info': {
        'num_satellites': len(env.satKeys),
        'num_sensors': len(env.sensorKeys)
    }
}

print(f'\nSaving model to {file_prefix}.pkl')
with open(file_prefix + '.pkl', 'wb') as f:
    dill.dump(save_data, f)


    
# Plot rolling average of rewards
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'))
plt.title('Rolling Average Reward (window=100)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')

# Plot rolling average of lost satellites
plt.subplot(1, 2, 2)
plt.plot(np.convolve(lost_satellites_history, np.ones(100)/100, mode='valid'), color='red')
plt.title('Rolling Average Lost Satellites (window=100)')
plt.xlabel('Episode')
plt.ylabel('Average Lost Satellites')

plt.tight_layout()
plt.savefig(file_prefix + '_training.png')
plt.close()
