from skyfield.api import load
from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.environment.Manuever import Maneuver
from SSN_RL.environment.AgentWrapper import AgentWrapper
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
A1 = AgentWrapper(allSatNames, [MHR.name])
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
    # --> update catalog 
    # TODO 
    # ii. last seen 
    lastSeen = C.lastSeen_mins(cTime)

    # TODO punish/reward: last seen and arriving events 
    rewardOrCost = 0

    # 3. get agent's responses
    action = A1.decide(rewardOrCost, lastSeen)
    
    # 4. execute actions 
    # i. send new tasks to appropriate sensors (delays)
    for sat in action:
        if action[sat]: # if not do nothing (i.e. False)
            sensors[action[sat]].sendSensorTask(cTime, sat, C.currentCatalog[sat], sConfigs)
    # ii. execute or continue executing tasks that have arrived to sensor
    for sensor in sensors:
        sensors[sensor].executeTasking(cTime)
    
    # iterate
    cTime += sConfigs.timeDelta

