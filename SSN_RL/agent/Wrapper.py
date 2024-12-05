from SSN_RL.environment.Sensor import SensorResponse
import numpy as np

class AgentWrapper:
    def __init__(self, agentID, assignedSats, availableSensors):
        self.agentID = agentID
        self.assignedSats = assignedSats
        self.availableSensors = availableSensors
        self.nSensors = len(self.availableSensors)
        self.nSats = len(self.assignedSats)
        
      
        # Initialize satellite index mapping
        self.sat2idx = {sat: idx for idx, sat in enumerate(self.assignedSats)}
        
        # Track lost satellites
        self.lostSatellites = set()
        self.cumulativeRewardOrCost = 0

    
    