
import torch
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.scenarioBuilder.scenarios import Scenario2Environment, Scenario2Environment_generalization_test
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.agent.DQNAgent import DQNAgent
from SSN_RL.agent.TrainingSpecs import TrainingSpecs
from SSN_RL.environment.rewards import reward_v1
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch
import datetime
import json

env = Scenario2Environment_generalization_test()

satKeys = env.satKeys
sensorKeys = env.sensorKeys

nSensors = env.nSensors
nSats = env.nSats
input_dim = nSats*2
output_dim = nSats


agent = DQNAgent("agent1", env.satKeys,env.sensorKeys)
agent.model.load_state_dict(torch.load("./scripts/experiments/DQN/dqn_scenario2_1.pth", weights_only=True))

file_prefix = './scripts/experiments/DQN/dqn_scen2'


agentID = "agent1"
sat2idx = {sat: idx for idx, sat in enumerate(satKeys)}
RESULTS = {}

REWARDS = []
COMPLETED_TASKS = []
MAN_DET = []
DROPPED_SCHED = []
INVALID_1 = []
INVALID_2 = []
TS = TrainingSpecs()
TS.num_episodes=100
epsilon = .1
start = datetime.datetime.now()
for episode in range(TS.num_episodes):
    total_reward = 0
    t, events, stateCat, Done = env.reset(deltaT=60*5)
    state = agent.encodeState(t, stateCat)
    while not Done:
        action, actions_decoded = agent.decide_on_policy(t, events, stateCat)
        

        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        t, events, stateCat, Done = env.step(actions_decoded)

        next_state = agent.encodeState(t, stateCat)
        
        start = datetime.datetime.now()
        agent.step(state, action, reward, next_state, Done)
        #print('Total time:',str((datetime.datetime.now() - start).total_seconds()), ' [s]')

        state = next_state
        total_reward += reward
    
    # decay epsilon 
    epsilon = max(TS.min_epsilon, epsilon * TS.epsilon_decay)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{TS.num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
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
# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -

env.debug_ec.display()
nManuevers = 0
for satKey in env.satTruth:
    nManuevers+= len(env.satTruth[satKey].maneuverList)
print("actual unique maneuvers: "+str(nManuevers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))
print(total_reward)


fig, ax = plt.subplots()


def string_to_color(s):

    np.random.seed(len(s))  # Set the seed to ensure consistency for the same string
    
    return np.random.rand(3,) 
c = ["black", "#29A634", "#D1980B", "#D33D17", "#9D3F9D", "#00A396", "#DB2C6F", "#8EB125", "#946638", "#7961DB"]
colors = {}
for i in range(len(env.satKeys)):
    colors[env.satKeys[i]]=c[i]






sEpoch = env.sConfigs.scenarioEpoch

i = 0
for sensor in env.sensorMap:
    for task in env.sensorMap[sensor].completedTasks:
        ax.plot([hrsAfterEpoch(sEpoch, task.startTime),hrsAfterEpoch(sEpoch,task.stopTime) ], [i, i], color=colors[task.satID], linewidth=4)
    i+=1
for satKey in env.satTruth:
    for m in env.satTruth[satKey].maneuverList:
        ax.axvline(x=hrsAfterEpoch(sEpoch,m.time), color=colors[satKey], linestyle=':', linewidth=2)
for event in env.debug_uniqueManeuverDetections:
    ax.axvline(x=hrsAfterEpoch(sEpoch,event.arrivalTime), color=colors[event.satID], linestyle='-.', linewidth=2)


plt.ylabel('Executed Schedule at Sensor (MHR)')
plt.xlabel('time [hours after scenario epoch]')
plt.show()