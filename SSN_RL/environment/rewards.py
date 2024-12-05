import numpy as np
from SSN_RL.environment.Sensor import SensorResponse
def reward_v1(t, events, stateCatalog, agentID, sat2idx):
    rewardOrCost = 0
    
    for event in events:
        if event.agentID == agentID:
            
            if event.type == SensorResponse.INVALID:
                rewardOrCost -= 50
            if event.type == SensorResponse.INVALID_TIME:
                rewardOrCost -= 5
            elif event.type == SensorResponse.DROPPED_SCHEDULING:
                rewardOrCost -= 1
            elif event.type == SensorResponse.DROPPED_LOST:
                rewardOrCost -= 1000
            elif event.type == SensorResponse.COMPLETED_NOMINAL:
                rewardOrCost += 15
            elif event.type == SensorResponse.COMPLETED_MANEUVER:
                rewardOrCost += 50
    
    lastSeen = np.array([
        stateCatalog.lastSeen_mins(t, sat) 
        for sat in sat2idx.keys()
    ])
    
    rewardOrCost -= np.sum(lastSeen > 90) * 2
    
    return rewardOrCost

