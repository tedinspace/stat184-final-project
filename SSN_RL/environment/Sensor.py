from SSN_RL.environment.Messages import PendingTaskMessage, EventMessage, TaskCommand
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

class SensorState(Enum):
    RESTING = 1,
    SLEWING = 2,
    TRACKING = 3

class Sensor:
    def __init__(self, name, lla):
        self.name = name
        self.modality = SensorModality.UNSPEC
        #self.currentState = SensorState.RESTING
        self.activeTask = False

        self.groundObserver = wgs84.latlon(lla[0], lla[1], lla[2])

        self.pendingIncomingTasks = []
        self.scheduledTasks = []
        self.pendingOutgoingInformation = []

        self.taskDelay = 7.5 # [mins] - time for message to be received by sensor from agent 
        self.taskDelayRand = [-5,5] # [-mins, +mins] randomness added tasking delay 
        self.responseDelay = 5 ; 
        self.responseDelayRand = [-2,4]   
        self.catalogUpdateDelay = 5 ; 
        self.catalogUpdateDelayRand = [-2,2]  

        self.slewAndSettleTimeMins = 1.5 # [mins]
        self.scheduleAheadLimitMins = 30 # [mins]
        self.taskingLengthMins = 5.5 
        


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

    def canAcquire_plus_crystalBall(self, t, truthSatInfo ):

        if truthSatInfo.maneuveredBetween(self.activeTask.availableState.epoch, t): 
            print("maneuver detected "+self.activeTask.satID)
            self.activeTask.maneuverDetected = True
        elif truthSatInfo.maneuveredBetween(self.activeTask.availableState.epoch, self.activeTask.stopTime): 
            print("maneuver detected during tracking "+self.activeTask.satID)
            self.activeTask.maneuverDetected = True
        else: 
            print("nominal "+self.activeTask.satID)
        
        return True

        


    def tick(self,t,truthStates):
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
        remainingPendingTasks = []
        sentInvalidState = []
        for taskExe in tasksToExecute:
            if self.isVisible(taskExe.availableState.at(t), t):
                remainingPendingTasks.append(taskExe)
            else:
                # state not visible
                sentInvalidState.append(taskExe)
                self.pendingOutgoingInformation.append(EventMessage(SensorResponse.INVALID, t + (self.responseDelay + random.uniform(self.responseDelayRand[0], self.responseDelayRand[1]) )/MPD, taskExe,[] ))
        # 3. 
        if self.activeTask:
            if self.activeTask.checkedAcquisition == False:
                # haven't acquired yet on active tasking yet
                if t > self.activeTask.startTime:

                    if(self.canAcquire_plus_crystalBall(t, truthStates[self.activeTask.satID])):
                        # can acquire
                        self.activeTask.checkedAcquisition = True # mark as such
                    else:
                        # can't acquire
                        self.pendingOutgoingInformation.append(EventMessage(SensorResponse.DROPPED_LOST, t + (self.responseDelay + random.uniform(self.responseDelayRand[0], self.responseDelayRand[1]) )/MPD, self.activeTask.taskMessage, [] ))                
                        self.activeTask = False 
            
        if self.activeTask and self.activeTask.stopTime < t:
            # - is there an active task that's done and ready to be sent out
            if self.activeTask.maneuverDetected:
                #print(self.activeTask.satID)
                #print(self.activeTask.satID)
                #print("MANUEVER")
                # - maneuver detected state
                self.pendingOutgoingInformation.append(EventMessage(SensorResponse.COMPLETED_MANEUVER, t + (self.catalogUpdateDelay + random.uniform(self.catalogUpdateDelayRand[0], self.catalogUpdateDelayRand[1]) )/MPD, self.activeTask.taskMessage, truthStates[self.activeTask.satID].activeObject ))
                self.activeTask = False 
            else:
                #print(self.activeTask.satID)
                #print("NOMINAL")
                # - no maneuver detected
                nState = self.activeTask.availableState
                nState.epoch = self.activeTask.stopTime
                self.pendingOutgoingInformation.append(EventMessage(SensorResponse.COMPLETED_NOMINAL, t + (self.catalogUpdateDelay + random.uniform(self.catalogUpdateDelayRand[0], self.catalogUpdateDelayRand[1]) )/MPD, self.activeTask.taskMessage, nState ))
                #print("stopped --> "+self.activeTask.satID)
                #print(t.utc_iso())
                self.activeTask = False
                
        if self.activeTask==False:
            # recheck for active task (it could have closed in previous step)
            if len(self.scheduledTasks) > 0:
                # take next scheduled task
                self.activeTask =self.scheduledTasks.pop()
                
            elif len(remainingPendingTasks)>0:
                # take next pending task if nothing in scheduled task list 
                nextTask = remainingPendingTasks.pop()
                self.activeTask = TaskCommand(nextTask, t+ self.slewAndSettleTimeMins/MPD, self.taskingLengthMins)
            #if self.activeTask:
            #    print("started --> "+self.activeTask.satID)
            #    print(t.utc_iso())

        # 4. schedule remaining tasks (or try)

        for taskExe in remainingPendingTasks:
            if len(self.scheduledTasks) ==0:
                # case there are no currently scheduled tasks
                self.scheduledTasks.append(TaskCommand(taskExe, self.activeTask.stopTime + self.slewAndSettleTimeMins/MPD, self.taskingLengthMins ))
            else:
                # try to schedule in accordance with other scheduled tasks
                requestedStartTime = self.scheduledTasks[-1].stopTime + self.slewAndSettleTimeMins/MPD
                if requestedStartTime < t + self.scheduleAheadLimitMins/MPD:
                    #  task is within the schedule ahead time
                    self.scheduledTasks.append(TaskCommand(taskExe, requestedStartTime, self.taskingLengthMins ))
                else:
                    # task not over the schedule ahead time 
                    self.pendingOutgoingInformation.append(EventMessage(SensorResponse.DROPPED_SCHEDULING, t + (self.responseDelay + random.uniform(self.responseDelayRand[0], self.responseDelayRand[1]) )/MPD, taskExe, [] ))








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



