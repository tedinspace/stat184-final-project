from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment
from SSN_RL.agent.LinearAgent import LinearQAgent
from SSN_RL.environment.rewards import reward_v1
import numpy as np
import datetime
import dill
import os
import matplotlib.pyplot as plt

# Configuration
NUM_TEST_EPISODES = 100
model_path = './scripts/experiments/Linear/toy2_linear_v1.pkl'
results_prefix = './scripts/experiments/Linear/toy2_linear_v1_test'

# Load trained model
print(f"Loading model from {model_path}")
with open(model_path, 'rb') as f:
    saved_data = dill.load(f)

# Initialize environment and agent
env = Scenario2Environment()
agent = LinearQAgent(
    agentID="agent1",
    assigned_sats=env.satKeys,
    assigned_sensors=env.sensorKeys,
    epsilon=0.0  # No exploration during testing
)

# Load trained weights
agent.weights = saved_data['weights']

# Testing metrics
test_rewards = []
test_episode_lengths = []
test_lost_satellites = []
action_distribution = []

print(f"\nTesting model on {NUM_TEST_EPISODES} episodes...")
start = datetime.datetime.now()

for episode in range(NUM_TEST_EPISODES):
    total_reward = 0
    steps = 0
    lost_satellites = 0
    episode_actions = []
    
    # Reset environment and agent
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    
    while not Done:
        # Take action
        actions, actions_decoded = agent.decide(t, events, stateCat)
        episode_actions.extend(actions)
        
        # Get reward and advance environment
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        t, events, stateCat, Done = env.step(actions_decoded)
        
        # Update metrics
        total_reward += reward
        steps += 1
        
        # Track lost satellites
        for event in events:
            if event.agentID == agent.agentID and hasattr(event, 'type'):
                if event.type == 'DROPPED_LOST':
                    lost_satellites += 1
    
    # Record episode metrics
    test_rewards.append(total_reward)
    test_episode_lengths.append(steps)
    test_lost_satellites.append(lost_satellites)
    action_distribution.extend(episode_actions)
    
    if (episode + 1) % 10 == 0:
        print(f"Completed {episode + 1}/{NUM_TEST_EPISODES} test episodes")

# Calculate test statistics
test_time = (datetime.datetime.now() - start).total_seconds()
test_results = {
    'average_reward': np.mean(test_rewards),
    'std_reward': np.std(test_rewards),
    'average_episode_length': np.mean(test_episode_lengths),
    'average_lost_satellites': np.mean(test_lost_satellites),
    'action_distribution': np.bincount(np.array(action_distribution) + 1),  # +1 to handle -1 actions
    'test_time': test_time
}

# Print test results
print("\nTest Results:")
print(f"Average Reward: {test_results['average_reward']:.2f} ± {test_results['std_reward']:.2f}")
print(f"Average Episode Length: {test_results['average_episode_length']:.2f}")
print(f"Average Lost Satellites: {test_results['average_lost_satellites']:.2f}")
print(f"Test Time: {test_time:.2f} seconds")

# Save test results
print(f"\nSaving test results to {results_prefix}_results.pkl")
with open(results_prefix + '_results.pkl', 'wb') as f:
    dill.dump(test_results, f)

    
# Create figure with subplots
plt.figure(figsize=(15, 5))

# Plot reward distribution
plt.subplot(1, 3, 1)
plt.hist(test_rewards, bins=20)
plt.title('Test Reward Distribution')
plt.xlabel('Reward')
plt.ylabel('Count')

# Plot episode lengths
plt.subplot(1, 3, 2)
plt.hist(test_episode_lengths, bins=20)
plt.title('Episode Length Distribution')
plt.xlabel('Steps')
plt.ylabel('Count')

# Plot action distribution
plt.subplot(1, 3, 3)
actions = np.arange(-1, len(env.sensorKeys))
plt.bar(actions, test_results['action_distribution'])
plt.title('Action Distribution')
plt.xlabel('Action')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig(results_prefix + '_analysis.png')
plt.close()
