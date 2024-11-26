from skyfield.api import load
from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.environment.Manuever import Maneuver
from SSN_RL.environment.Agent import AgentWrapper, AgentRewardCostTracker
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR


# init configs (epoch + length)
sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 4.5)

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

cTime = sConfigs.scenarioEpoch
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
        # TODO configure rewards
        if event.type == SensorResponse.COMPLETED_NOMINAL:
            # --> update catalog (reward)
            RC.reward(event.agentID, 10) # TODO
        elif event.type == SensorResponse.COMPLETED_MANEUVER:
            # --> update catalog (rewardx2)
            RC.reward(event.agentID, 20) # TODO
        elif event.type == SensorResponse.INVALID:
            # --> punish invalid 
            RC.reward(event.agentID, -15) # TODO
        elif event.type == SensorResponse.DROPPED_SCHEDULING:
            # --> punish dropped? 
            RC.reward(event.agentID, 0) # TODO 
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
    # ii. execute or continue executing tasks that have arrived to sensor
    for sensor in sensors:
        sensors[sensor].executeTasking(cTime)
    
    # iterate
    cTime += sConfigs.timeDelta

