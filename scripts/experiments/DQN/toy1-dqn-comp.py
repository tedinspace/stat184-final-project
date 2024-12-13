
import torch
from SSN_RL.agent.algorithms.trivial import randomAction,noAction
from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1,ToyEnvironment1_generalization_test_1
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.agent.functions.encode import encode_basic_v1
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.environment.rewards import reward_v1
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
from SSN_RL.environment.Sensor import SensorResponse
import json
import numpy as np

n_runs = 100
env = ToyEnvironment1_generalization_test_1()

satKeys = env.satKeys
sensorKeys = env.sensorKeys

nSensors = env.nSensors
nSats = env.nSats
input_dim = nSats*2
output_dim = nSats


agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
agent.model.load_state_dict(torch.load("./scripts/experiments/DQN/dqn_toy1_v1.pth", weights_only=True))

file_prefix = './scripts/experiments/DQN/dqn_toy1_comp_gen'

RESULTS = {}

agentID = "agent1"


DQN_rewards = []

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}
for episode in range(n_runs):
    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    state = encode_basic_v1(t, events, stateCat, agentID, sat2idx)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    while not Done:
        #action = randomAction(nSensors, nSats)
        action = agent.decide_on_policy_inner(state)
        action_spec =  np.round(action.numpy()).astype(int)
        t, events, stateCat, Done = env.step({agent.agentID: decodeActions(action_spec, agent.assigned_sats, agent.assigned_sensors)})
        next_state = encode_basic_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) 
        reward =  reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)

        state = next_state
        total_reward += reward
    REWARDS.append(float(total_reward))
    COMPLETED_TASKS.append(float(env.debug_ec.eventCounts[SensorResponse.COMPLETED_MANEUVER]+ env.debug_ec.eventCounts[SensorResponse.COMPLETED_NOMINAL]))
    DROPPED_SCHED.append(float(env.debug_ec.eventCounts[SensorResponse.DROPPED_SCHEDULING]))
    MAN_DET.append(float(env.debug_ec.eventCounts[SensorResponse.UNIQUE_MAN]/env.countUniqueManeuvers()))

    INVALID_1.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID]))
    INVALID_2.append(float(env.debug_ec.eventCounts[SensorResponse.INVALID_TIME]))

RESULTS["DQN"]={
    "rewards": REWARDS,
    "completed": COMPLETED_TASKS, 
    "dropped": DROPPED_SCHED,
    "man_det": MAN_DET,
    "invalid": INVALID_1, 
    "invalid_time": INVALID_2
}



with open(file_prefix+"_comp_gen.json", 'w') as f:
    json.dump(RESULTS, f)