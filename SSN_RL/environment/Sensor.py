from SSN_RL.environment.Messages import PendingTaskMessage, EventMessage
from enum import Enum
from SSN_RL.utils.time import MPD
from skyfield.api import wgs84, load
import random


eph = load('de421.bsp')
sun = eph['sun']-eph['earth']

class SensorModality(Enum):
    UNSPEC = 1 # unspecified modality; treated like RADAR
    RADAR  = 2 # RADAR
    OPTICS = 3 # Optical Telescope

class SensorResponse(Enum):
    INVALID = 1 # agent asked sensor to task on state it can't see
    DROPPED_SCHEDULING = 2 # sensor can't schedule task 
    DROPPED_LOST = 3 # sensor tried to look at object, but it wasn't there
    COMPLETED_NOMINAL = 4 # sensor completed tasking, nothing occurred 
    COMPLETED_MANEUVER = 5 # sensor completed tasking and discovered a maneuver

class Sensor:
    def __init__(self, name, lla):
        self.name = name
        self.lat = lla[0]
        self.lon = lla[1]
        self.alt = lla[2]
        self.modality = SensorModality.UNSPEC
        self.groundObserver = wgs84.latlon(self.lat, self.lon, self.alt)
        self.pendingIncomingTasks = []
        self.executingTasks = []
        self.pendingOutgoingInformation = []
        self.taskDelay = 7.5 # [mins] - time for message to be received by sensor from agent 
        self.taskDelayRand = [-5,5] # [-mins, +mins] randomness added tasking delay 
        self.responseDelay = 5 ; 
        self.responseDelayRand = [-2,4]   
        self.catalogUpdateDelay = 5 ; 
        self.catalogUpdateDelayRand = [-2,2]  

    def updateModality(self, modeEnum):
        '''change from unspecified to RADAR or Optics'''
        self.modality = modeEnum; 

    def hasLineOfSight(self, X, t):
        '''Point to the sky, is it there?'''
        alt, _, _ = (X-self.groundObserver.at(t)).altaz()
        return alt.degrees > 5

    def isVisible(self, X, t):
        '''for RADARs same as hasLineOfSight (with less generous horizon); 
           for optics calcuates if it is day at sensor          '''
        alt, _, _ = (X-self.groundObserver.at(t)).altaz()
        if alt.degrees > 10:
            if self.modality == SensorModality.OPTICS:
                # optics logic 
                alt_solar, _, _ = (sun.at(t)-self.groundObserver.at(t)).altaz()
                return alt_solar.degrees < -12 # up until nautical twilight starts
            return True 
        return False  
    
    def sendSensorTask(self, t, agentID, satName, currentCatalogState ):
        '''stages a task message'''
        self.pendingIncomingTasks.append(PendingTaskMessage(agentID, satName,t + (self.taskDelay + random.uniform(self.taskDelayRand[0], self.taskDelayRand[1]) )/MPD, currentCatalogState ))

    def executeTasking(self,t):
        '''checks if messages has arrived, if so starts executing them'''
        tasksToExecute = []
        tasksStillPending = []
        # 1. see if pending task has reached the sensor or not yet
        for taskMessage in self.pendingIncomingTasks:
            if taskMessage.arrivalTime < t:
                # arrived: if arrival time is in the past
                tasksToExecute.append(taskMessage)
            else:
                # hasn't arrived: add back onto pending list
                tasksStillPending.append(taskMessage)
        self.pendingIncomingTasks = tasksStillPending

        # 2. make sure that the state sent to sensor is valid
        sentInvalidState = []
        for taskExe in tasksToExecute:
            if not self.isVisible(taskExe.availableState.at(t), t):
                x = 0
            else:
                sentInvalidState.append(taskExe)
                self.pendingOutgoingInformation.append(EventMessage(SensorResponse.INVALID, t + (self.responseDelay + random.uniform(self.responseDelayRand[0], self.responseDelayRand[1]) )/MPD, taskExe ))
              


    def checkForUpdates(self, t):
        '''check if response info is ready to be received by agent/environment'''
        outGoing = []
        stillPending = []
        for info in self.pendingOutgoingInformation:
            if info.arrivalTime < t:
                outGoing.append(info)
            else:
                stillPending.append(info)
        self.pendingOutgoingInformation = stillPending
        return outGoing



