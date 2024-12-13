from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1,ToyEnvironment1_generalization_test_1
from SSN_RL.agent.QAgent import QAgent
from SSN_RL.agent.HeuristicAgent import HeuristicAgent
from SSN_RL.agent.algorithms.trivial import randomAction, noAction
from SSN_RL.environment.rewards import reward_v1
from SSN_RL.agent.functions.decode import decodeActions

import datetime
import dill
import json
import numpy as np

start = datetime.datetime.now()
file_prefix = './scripts/experiments/QLearning/ql_toy1_v2'

RESULTS = {}

<<<<<<< HEAD
EPISODES = 100
=======
EPISODES = 1000
>>>>>>> chirag--linear_agent


env = ToyEnvironment1_generalization_test_1()
agent = QAgent("agent1", env.satKeys, env.sensorKeys)
with open(file_prefix+'.pkl', 'rb') as f:
    loaded_qTable = dill.load(f)
    agent.qTable = loaded_qTable

Q_REWARDS = []
Q_COMPLETED_TASKS = []
Q_MAN_DET = []
Q_DROPPED_SCHED = []
for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    state = agent.encodeState(t, stateCat)
    state_disc = agent.discretizeState(state)


    while not Done:
        # take actions
        action, actions_decoded, action_disc = agent.decide_on_policy(t, events, stateCat)

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
        

    Q_REWARDS.append(float(total_reward))
    Q_COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    Q_MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/2))
    Q_DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["Q"]={
    "rewards": Q_REWARDS,
    "completed": Q_COMPLETED_TASKS, 
    "dropped": Q_DROPPED_SCHED,
    "man_det": Q_MAN_DET,
}

agent = HeuristicAgent("agent1", env.satKeys,env.sensorKeys)

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()


    while not Done:
        action = agent.decide(t, events, stateCat)
        t, events, stateCat, Done = env.step({agent.agentID: action})
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        

    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/2))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["heuristic"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
}


REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)


    while not Done:
        action = randomAction(env.nSensors, env.nSats)
        t, events, stateCat, Done = env.step({agent.agentID:  decodeActions(action, env.satKeys, env.sensorKeys)})
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        

    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/2))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["random"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
}

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
for episode in range(EPISODES):

    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)


    while not Done:
        action = noAction( env.nSats)
        t, events, stateCat, Done = env.step({agent.agentID:  decodeActions(action, env.satKeys, env.sensorKeys)})
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        

    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/2))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["no_action"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
}



with open(file_prefix+"_comp_gen.json", 'w') as f:
    json.dump(RESULTS, f)