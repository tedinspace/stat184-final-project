import random


class AgentWrapper:
    def __init__(self, agentID, assignedSats, availableSensors):
        self.agentID = agentID
        self.assignedSats = assignedSats # list of names/keys
        self.availableSensors = availableSensors # list of names/keys
        # TODO - task type headcount, medium, long task (0, 1, 2)? 
        self.nSensors = len(self.availableSensors)

    def decide(self,rewardOrCost, lastSeenCatalog):
        decisions = {}
        for sat in self.assignedSats:
            randomChoice = random.randint(0, self.nSensors)
            if randomChoice == 0:
                decisions[sat] = False
            else:
                decisions[sat]=self.availableSensors[randomChoice-1]
           
        return decisions

class AgentRewardCostTracker:
    def __init__(self, agentIDs):
        self.rewardPunishments_thisRound = {}
        self.rewardPunishments_allTime = {}
        for agentID in agentIDs:
            self.rewardPunishments_thisRound[agentID]=0
            self.rewardPunishments_allTime[agentID]=0
    def resetForRound(self):
        for agentID in self.rewardPunishments_thisRound:
            self.rewardPunishments_thisRound[agentID]=0
    def reward(self,agentID,  rewardOrCost):
        self.rewardPunishments_thisRound[agentID]+= rewardOrCost
        self.rewardPunishments_allTime[agentID]+= rewardOrCost
