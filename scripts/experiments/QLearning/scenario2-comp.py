from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment_generalization_test
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
file_prefix = './scripts/experiments/QLearning/scenario2_v2'

RESULTS = {}

EPISODES = 100


env = Scenario2Environment_generalization_test()
agent = QAgent("agent1", env.satKeys, env.sensorKeys)
with open(file_prefix+'.pkl', 'rb') as f:
    loaded_qTable = dill.load(f)
    agent.qTable = loaded_qTable

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
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
        

    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/2))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))
    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["Q"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}

agent = HeuristicAgent("agent1", env.satKeys,env.sensorKeys)

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
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
    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["heuristic"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}


REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
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
    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["random"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
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
    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")


RESULTS["no_action"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}



with open(file_prefix+"_comp_gen.json", 'w') as f:
    json.dump(RESULTS, f)