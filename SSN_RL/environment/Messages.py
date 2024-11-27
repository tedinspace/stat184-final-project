from SSN_RL.utils.time import MPD
class PendingTaskMessage:
    '''message type that is sent to sensor'''
    def __init__(self,agentID, satID, arrivalTime, availableState):
        self.agentID = agentID # agent that sent message
        self.satID = satID # satellite
        self.arrivalTime = arrivalTime # time it gets to sensor
        self.availableState = availableState # state in catalog when message was sent


class TaskCommand:
    '''internal message that handles executing (or scheduled) tasks within the sensor; these are never sent out'''
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
    '''All event messages go here '''
    def __init__(self,type, arrivalTime, taskMessage, newState):
        self.type  = type # SensorResponseType
        self.arrivalTime = arrivalTime # time message will get to agent
        self.agentID = taskMessage.agentID # agent who opened the loop
        self.satID = taskMessage.satID # satellite the message was about
        self.newState = newState # sometimes empty array; only contains a state if a new state was estimated by sensor
        self.crystalBallState = [] # 
        #self.taskMessage = taskMessage
