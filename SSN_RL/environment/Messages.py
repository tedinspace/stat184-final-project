from SSN_RL.utils.time import MPD
class PendingTaskMessage:
    def __init__(self,agentID, satID, arrivalTime, availableState):
        self.agentID = agentID
        self.satID = satID
        self.arrivalTime = arrivalTime
        self.availableState = availableState


class TaskCommand:
    def __init__(self,taskMessage, startTime, lengthMins):
        self.agentID = taskMessage.agentID
        self.satID = taskMessage.satID
        self.arrivalTime = taskMessage.arrivalTime
        self.availableState = taskMessage.availableState
        self.startTime = startTime
        self.stopTime = startTime + lengthMins/MPD
        self.maneuverDetected = False
        self.taskMessage = taskMessage
        self.checkedAcquisition = False
    
class EventMessage:
    def __init__(self,type, arrivalTime, taskMessage, newState):
        self.type  = type
        self.arrivalTime = arrivalTime
        self.agentID = taskMessage.agentID
        self.satID = taskMessage.satID
        self.newState = newState
        #self.taskMessage = taskMessage
