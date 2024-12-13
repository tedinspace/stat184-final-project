from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment_generalization_test
from SSN_RL.agent.LinearAgent import LinearQAgent
from SSN_RL.environment.rewards import reward_v1
import numpy as np
import datetime
import dill
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
from SSN_RL.environment.Sensor import SensorResponse
import json

EPISODES = 100
file_prefix = './scripts/experiments/Linear/toy2_linear_v1'
model_path = './scripts/experiments/Linear/toy2_linear_v1.pkl'

start = datetime.datetime.now()


with open(model_path, 'rb') as f:
    saved_data = dill.load(f)
env = Scenario2Environment_generalization_test()
agent = LinearQAgent(
    agentID="agent1",
    assigned_sats=env.satKeys,
    assigned_sensors=env.sensorKeys,
    epsilon=0.0  # No exploration during testing
)
RESULTS = {}

print(env.countUniqueManeuvers())

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
for episode in range(EPISODES):
    total_reward = 0
    steps = 0
    
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    agent.epsilon=0
    agent.eps_threshold =0

    state = agent.encodeState(t, stateCat)
    
    while not Done:
        actions, actions_decoded = agent.decide(t, events, stateCat)
        
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        
        t, events, stateCat, Done = env.step(actions_decoded)
        steps += 1
    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/env.countUniqueManeuvers()))

    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))
    

env.satTruth
RESULTS["LQ"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}
with open(file_prefix+"_comp_gen.json", 'w') as f:
    json.dump(RESULTS, f)