from SSN_RL.environment.Sensor import SensorResponse
import numpy as np

class AgentWrapper:
    def __init__(self, agentID, assignedSats, availableSensors):
        self.agentID = agentID
        self.assignedSats = assignedSats # list of names/keys
        self.availableSensors = availableSensors # list of names/keys
        # TODO - task type headcount, medium, long task (0, 1, 2)? 
        self.nSensors = len(self.availableSensors)
        self.nSats = len(self.assignedSats)
        self.sat2idx = {}
        for i in range(self.nSats):
            self.sat2idx[self.assignedSats[i]]=i
        self.lostSatellites = set()
        self.cumulativeRewardOrCost = 0

    def decide(self, t, events, stateCatalog):

        encodedState, rewardOrCost = self.encodeCurrentState(t, events, stateCatalog)
        self.cumulativeRewardOrCost+= rewardOrCost

        # RL HAPPENS HERE
        # takes in encodedState and rectures actions 
        # actions = RL.decision(encodedState)
        
        # here are random actions each position is a satellite 
        actions = np.random.randint(low=-1, high=self.nSensors, size=self.nSats)

        # decode actions for environment to use
        decisions = {}
        for i in range(self.nSats):
            a = actions[i]
            sat = self.assignedSats[i]

            if a == -1:
                decisions[sat] = False
            else:
                decisions[sat]=self.availableSensors[a]               
           
        return decisions
    

    def encodeCurrentState(self, t, events, stateCatalog):

        eventEncoding = np.zeros(self.nSats)
        rewardOrCost = 0
        for event in events:
            if event.agentID == self.agentID:
                if event.type == SensorResponse.INVALID:
                    rewardOrCost += -5
                elif event.type == SensorResponse.DROPPED_SCHEDULING:
                    rewardOrCost += -1 
                elif event.type == SensorResponse.DROPPED_LOST:
                    eventEncoding[self.sat2idx[event.satID]]=2
                    rewardOrCost += -100
                    self.lostSatellites.add(event.satID)
                elif rewardOrCost == SensorResponse.COMPLETED_NOMINAL:
                    rewardOrCost += 15
                elif rewardOrCost == SensorResponse.COMPLETED_MANEUVER:
                    eventEncoding[self.sat2idx[event.satID]]=1
                    rewardOrCost += 50

        lastSeen = np.zeros(self.nSats)
        for satKey in self.assignedSats:
            lastSeen[self.sat2idx[satKey]]=stateCatalog.lastSeen_mins(t, satKey)
            # TODO punish states for being too old? 
        
        # TODO punish lost satellites each round? 
        # for s in self.lostSatellites: 

        return np.concatenate((eventEncoding, lastSeen)), rewardOrCost


        



