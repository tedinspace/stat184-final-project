
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.algorithms.trivial import randomAction, noAction
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.rewards import reward_v1
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1

env = ToyEnvironment1()
nSensors = env.nSensors
nSats = env.nSats

satKeys = env.satKeys
sensorKeys = env.sensorKeys

agentID = "agent1"
sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}


TS = TrainingSpecs()
TS.num_episodes=100
epsilon = .1

for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset()
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    
    while not Done:
        #action = randomAction(nSensors, nSats)
        action = noAction(nSats)

        t, events, stateCat, Done = env.step({agentID: decodeActions(action, satKeys, sensorKeys)})
        next_state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)

        reward =  reward_v1(t, events, stateCat, agentID, sat2idx)

        state = next_state
        total_reward += reward
    
    # decay epsilon 
    epsilon = max(TS.min_epsilon, epsilon * TS.epsilon_decay)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        

