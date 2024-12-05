
from SSN_RL.agent.QTableAgent import QTableAgent
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1

env = ToyEnvironment1()
nSensors = env.nSensors
nSats = env.nSats

satKeys = env.satKeys
sensorKeys = env.sensorKeys

A = [QTableAgent("agent 1", env.satKeys, env.sensorKeys)]

sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}


TS = TrainingSpecs()
TS.num_episodes=100
epsilon = .1

for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    
    # reset ev
    for agent in A:
        agent.reset()

    

    while not Done:
        actions = {}
        for agent in A:
            a, reward = agent.decide(t, events, stateCat)
            actions[agent.agentID]=a
        

        t, events, stateCat, Done = env.step(actions)


        total_reward += reward
    
    # decay epsilon 
    epsilon = max(TS.min_epsilon, epsilon * TS.epsilon_decay)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        

# TODO pickle/save QTable for reloading