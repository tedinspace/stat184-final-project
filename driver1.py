from skyfield.api import load
import matplotlib.pyplot as plt

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.environment.Manuever import Maneuver
from SSN_RL.environment.Agent import AgentWrapper
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR

from SSN_RL.debug.Loggers import EventCounter
EC = EventCounter()

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

# - catalog 
C = StateCatalog(S) # we are initializing with the truth states; this doesn't have to be the case

# scenario loop 
eventsFromSensor = []
Successful_Tasks = []
cTime = sConfigs.scenarioEpoch


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

    # ii. conduct catalog state updates 
    curratedEvents = [] # events we want to send to agent as a state
    for event in events:
        EC.increment(event.type)
        #print(event.satID + "-->"+ str(event.type))
        if event.type == SensorResponse.COMPLETED_NOMINAL:
            # --> update catalog 
            C.updateState(event.satID, event.newState)
            curratedEvents.append(event)
        elif event.type == SensorResponse.COMPLETED_MANEUVER:
            # --> update catalog with maneuver
            if not C.wasManeuverAlreadyDetected(cTime, event.satID, event.newState): 
                EC.increment("unique maneuver detection")
                uniqueManeuverDetections_states.append(event)  
            else:
                # if not unique, just credit as state update
                event.type = SensorResponse.COMPLETED_NOMINAL
            curratedEvents.append(event)

            C.updateState(event.satID, event.newState)
        elif event.type == SensorResponse.DROPPED_SCHEDULING:
            # --> dropped tasks
            curratedEvents.append(event)
        else: 
            curratedEvents.append(event)
            
    

    # 3. get agent's responses
    actions = {}
    for agent in A:
        actions[agent.agentID]=agent.decide(curratedEvents, C)
    
    # 4. execute actions 
    # i. send new tasks to appropriate sensors (delays)
    for agentKey in actions:
        for sat in actions[agentKey]:
            if actions[agentKey][sat]: # if not do nothing (i.e. False)
                sensors[actions[agentKey][sat]].sendSensorTask(cTime, agentKey, sat, C.currentCatalog[sat])
                EC.increment("tasks sent")
    # ii. execute or continue executing tasks that have arrived to sensor
    for sensor in sensors:
        sensors[sensor].tick(cTime, S)
    
    # iterate
    cTime += sConfigs.timeDelta




# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -
EC.display()


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

plt.show()

