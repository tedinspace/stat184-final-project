class PendingTaskMessage:
    def __init__(self,agentID, satID, arrivalTime, availableState):
        self.agentID = agentID
        self.satID = satID
        self.arrivalTime = arrivalTime
        self.availableState = availableState
    
class EventMessage:
    def __init__(self,type, arrivalTime, taskMessage):
        self.type  = type
        self.arrivalTime = arrivalTime
        self.agentID = taskMessage.agentID
        self.satID = taskMessage.satID
        #self.taskMessage = taskMessage
