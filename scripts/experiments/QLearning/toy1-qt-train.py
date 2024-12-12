from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1
import dill
from SSN_RL.agent.QAgent import QAgent
from SSN_RL.environment.rewards import reward_v1
import datetime
start = datetime.datetime.now()
file_prefix = './scripts/experiments/QLearning/ql_toy1_v2'

EPISODES = 10000


env = ToyEnvironment1()
agent = QAgent("agent1", env.satKeys, env.sensorKeys)


for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()

    state = agent.encodeState(t, stateCat)
    state_disc = agent.discretizeState(state)


    while not Done:
        # take actions
        action, actions_decoded, action_disc = agent.decide(t, events, stateCat)

        #print(state_disc)
        #print(action_disc)

        # get reward
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        # advance env
        t, events, stateCat, Done = env.step(actions_decoded)
        
        # next state
        next_state = agent.encodeState(t, stateCat)
        next_state_disc = agent.discretizeState(next_state)
        

        agent.updateQTable(state_disc, action_disc, reward, next_state_disc)
                

        state = next_state
        state_disc = next_state_disc

        total_reward += reward
    

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.eps_threshold:.4f}")


print('Total time:',str((datetime.datetime.now() - start).total_seconds()/60), 'mins')
print(total_reward)

#print("NOT SAVING FILE")
with open(file_prefix+".pkl", 'wb') as f:
    dill.dump(agent.qTable, f)


