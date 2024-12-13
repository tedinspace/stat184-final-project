from SSN_RL.scenarioBuilder.scenarios import ToyEnvironment1, ToyEnvironment1_generalization_test_1
from SSN_RL.agent.LinearAgent import LinearQAgent
from SSN_RL.environment.rewards import reward_v1
import numpy as np
import datetime
import dill
import matplotlib.pyplot as plt
from SSN_RL.utils.time import hrsAfterEpoch

EPISODES = 1
file_prefix = './scripts/experiments/Linear/linear_toy1_v1'
start = datetime.datetime.now()


env = ToyEnvironment1()
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


with open(file_prefix+'.pkl', 'rb') as f:
    loaded_data = dill.load(f)
    agent.weights = loaded_data["weights"]
    #agent.alpha = loaded_data['hyperparameters']['alpha']
    #agent.gamma = loaded_data['hyperparameters']['gamma']
    #agent.epsilon_dec = loaded_data['hyperparameters']['epsilon_dec']
    #agent.epsilon_min = loaded_data['hyperparameters']['epsilon_min']



for episode in range(EPISODES):
    total_reward = 0
    steps = 0
    
    t, events, stateCat, Done = env.reset(deltaT=5*60)
    agent.reset()
    agent.epsilon=0

    state = agent.encodeState(t, stateCat)
    
    while not Done:
        actions, actions_decoded = agent.decide(t, events, stateCat)
        
        reward = reward_v1(t, events, stateCat, agent.agentID, agent.sat2idx)
        total_reward += reward
        
        t, events, stateCat, Done = env.step(actions_decoded)
        steps += 1
    


# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -

env.debug_ec.display()
nManuevers = 0
for satKey in env.satTruth:
    nManuevers+= len(env.satTruth[satKey].maneuverList)
print("actual unique maneuvers: "+str(nManuevers))
print("scenario length "+str(env.sConfigs.scenarioLengthHours))
print("REWARD "+str(total_reward))
fig, ax = plt.subplots()
colors = {
    'MUOS1': 'blue', 
    'MUOS2': 'red',
    "MUOS3": 'orange', 
    "MUOS4": 'green',
    "MUOS5": 'purple', 
    'AEHF 1 (USA 214)': 'red', 
    'AEHF 2 (USA 235)': 'blue'

}

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


plt.title('Plotted Results of Randomized Actions')
plt.ylabel('Executed Schedule at Sensor (MHR)')
plt.xlabel('time [hours after scenario epoch]')
plt.show()