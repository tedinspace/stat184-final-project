from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.utils.time import defaultEpoch


def seenAndUnseenAtSensors(states, sensors):
    '''given a  list of 3LE states and sensors, see if there is line of sight access;
       should only be used for GEO states.
    '''
    t = defaultEpoch
    seen =  set()
    unseen = set()
    sConfigs = ScenarioConfigs(t, 10)
    stateMap = {}
    for state in states:
        sat =  Satellite(state[0], state[1], state[2], sConfigs )
        unseen.add(sat.name)
        stateMap[sat.name]=state
        for sensor in sensors:
            if sensor.hasLineOfSight(sat.activeObject.at(t), sConfigs.scenarioEpoch):
                seen.add(sat.name)
                if sat.name in unseen:
                    unseen.remove(sat.name)

    seenStates = []
    for sat in seen:
        seenStates.append(stateMap[sat])
    return list(seen), list(unseen), seenStates