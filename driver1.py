from skyfield.api import load
import matplotlib.pyplot as plt

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.environment.Manuever import Maneuver
from SSN_RL.environment.Agent import AgentWrapper, AgentRewardCostTracker
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR


# init configs (epoch + length)
sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 14.5)

# generate true satellites
S = {}
allSatNames = []
for i in [0, 2, 4]: # visibile by MHR
    muos_i = MUOS_CLUSTER[i]
    tmp =  Satellite(muos_i[0], muos_i[1], muos_i[2], sConfigs )
    tmp.reEpoch_init(60*i) # re-epoch 
    S[tmp.name] = tmp
    allSatNames.append(tmp.name)
# truth maneuver
S[allSatNames[0]].addManeuvers([Maneuver(10, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])
S[allSatNames[2]].addManeuvers([Maneuver(5.5,8.5, sConfigs)])

# init supporting objects ... 
# - sensors
sensors = {MHR.name: MHR}
# - agents 
A = [AgentWrapper("agent 1", allSatNames, [MHR.name])]
RC = AgentRewardCostTracker(["agent 1"])


# - catalog 
C = StateCatalog(S) # we are initializing with the truth states; this doesn't have to be the case

# scenario loop 
eventsFromSensor = []
Successful_Tasks = []
cTime = sConfigs.scenarioEpoch

tasksSent = 0
tasksDropped = 0
stateUpdates = 0
maneverDetections = 0
uniqueManeuverDetections = 0
uniqueManeuverDetections_states = []

while cTime < sConfigs.scenarioEnd:
    # 1. everything needs to move forward to the current time 
    # --> compute truth (and any maneuvers if necessary)
    for satKey in S:
        S[satKey].tick(cTime)
        

    # 2. gather information
    # i. check to see if any sensor events are available to agent 
    events = []
    for sensor in sensors:
        events += sensors[sensor].checkForUpdates(cTime)

    # ii. reset agent reward/punishment map
    RC.resetForRound()
    # ii. go through events
    for event in events:
        #print(event.satID + "-->"+ str(event.type))
        # TODO configure rewards
        if event.type == SensorResponse.COMPLETED_NOMINAL:
            # --> update catalog (reward)
            RC.reward(event.agentID, 10) # TODO
            C.updateState(event.satID, event.newState)
            stateUpdates+=1
        elif event.type == SensorResponse.COMPLETED_MANEUVER:
            # --> update catalog (rewardx2)
            if C.wasManeuverAlreadyDetected(cTime, event.satID, event.newState): 
                print("maneuver already detected; no credit for maneuver")
            else:
                uniqueManeuverDetections+=1
                uniqueManeuverDetections_states.append(event)
                RC.reward(event.agentID, 20) # TODO
            C.updateState(event.satID, event.newState)
            maneverDetections+=1
            stateUpdates+=1
        elif event.type == SensorResponse.INVALID:
            # --> punish invalid 
            RC.reward(event.agentID, -15) # TODO
        elif event.type == SensorResponse.DROPPED_SCHEDULING:
            # --> punish dropped? 
            RC.reward(event.agentID, 0) # TODO 
            tasksDropped+=1
        elif event.type == SensorResponse.DROPPED_LOST:
            # --> punish lost
            RC.reward(event.agentID, -10) # TODO
            
    
    # ii. last seen 
    lastSeen = C.lastSeen_mins(cTime)
    # TODO punish for too long

    # 3. get agent's responses
    actions = {}
    for agent in A:
        actions[agent.agentID]=agent.decide(RC.rewardPunishments_thisRound[agent.agentID], lastSeen)
    
    # 4. execute actions 
    # i. send new tasks to appropriate sensors (delays)
    for agentKey in actions:
        for sat in actions[agentKey]:
            if actions[agentKey][sat]: # if not do nothing (i.e. False)
                sensors[actions[agentKey][sat]].sendSensorTask(cTime, agentKey, sat, C.currentCatalog[sat])
                tasksSent+=1
    # ii. execute or continue executing tasks that have arrived to sensor
    for sensor in sensors:
        sensors[sensor].tick(cTime, S)
    
    # iterate
    cTime += sConfigs.timeDelta



print('---'*24)
print("> tasks sent - "+str(tasksSent))
print("> tasks dropped - "+str(tasksDropped))
print("> stateUpdates - "+str(stateUpdates))
print("> maneverDetections - "+str(maneverDetections))
print("> uniqueManeuverDetections - "+str(uniqueManeuverDetections))
print('---'*24)

print(len(sensors[MHR.name].completedTasks))
fig, ax = plt.subplots()

colors = {
    'MUOS1': 'blue', 
    "MUOS3": 'orange', 
    "MUOS5": 'purple'
}
for task in sensors[MHR.name].completedTasks:
    ax.plot([task.startTime.tt,task.stopTime.tt ], [1, 1], color=colors[task.satID], linewidth=2)
    
for satKey in S:
    for m in S[satKey].maneuverList:
        ax.axvline(x=m.time.tt, color=colors[satKey], linestyle=':', linewidth=2)
for event in uniqueManeuverDetections_states:
    ax.axvline(x=event.arrivalTime.tt, color=colors[event.satID], linestyle='-.', linewidth=2)



plt.grid(True)
plt.show()