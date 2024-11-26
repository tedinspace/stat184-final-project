import random


class AgentWrapper:
    def __init__(self, assignedSats, availableSensors):
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
