import numpy as np
from SSN_RL.environment.Sensor import SensorResponse

def encode_basic_v1(t, events, stateCatalog, agentID, sat2idx):
    
    eventEncoding = np.zeros(len(sat2idx))
    for event in events:
        if event.agentID == agentID:
            sat_idx = sat2idx[event.satID]
            if event.type == SensorResponse.COMPLETED_MANEUVER:
                eventEncoding[sat_idx] = 1
            elif event.type == SensorResponse.DROPPED_LOST:
                eventEncoding[sat_idx] = 2

        
    lastSeen = np.array([
        stateCatalog.lastSeen_mins(t, sat) 
        for sat in sat2idx.keys()
    ])
    return  np.concatenate((eventEncoding, lastSeen))


def encode_basic_v2(t, events, stateCatalog, agentID, sat2idx):
    
    eventEncoding = np.zeros(len(sat2idx))
    for event in events:
        if event.agentID == agentID:
            sat_idx = sat2idx[event.satID]
            if event.type == SensorResponse.DROPPED_SCHEDULING:
                eventEncoding[sat_idx] = 1
            if event.type == SensorResponse.INVALID_TIME:
                eventEncoding[sat_idx] = 2
            if event.type == SensorResponse.COMPLETED_NOMINAL:
                eventEncoding[sat_idx] = 3
            if event.type == SensorResponse.INVALID:
                eventEncoding[sat_idx] = 4
            if event.type == SensorResponse.COMPLETED_MANEUVER:
                eventEncoding[sat_idx] = 5
            if event.type == SensorResponse.DROPPED_LOST:
                eventEncoding[sat_idx] = 6

        
    lastSeen = np.array([
        stateCatalog.lastSeen_mins(t, sat) 
        for sat in sat2idx.keys()
    ])
    return  np.concatenate((eventEncoding, lastSeen))