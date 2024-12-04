import random
from skyfield.api import load

from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.Randomizer import Randomizer
from SSN_RL.debug.Loggers import EventCounter
from SSN_RL.utils.struct import list2map

class Environment: 
    def __init__(self, inputTleList, sensorList):
        self.R = Randomizer()
        # randomize epoch and scenario length
        self.sConfigs = self.R.randomizeScenarioSpecs().updateDT_careful(15)

        self.satTruth = self.R.randomizeSatTruth(inputTleList, self.sConfigs)
        self.sensorMap = list2map(sensorList)

        self.stateCatalog = StateCatalog(self.satTruth)

        self.t = self.sConfigs.scenarioEpoch

        # tabs 
        

        # save
        self.inputTleList = inputTleList
        self.inputSensorList = sensorList
        
        # debug
        self.debug_ec = EventCounter()

        self.debug_uniqueManeuverDetections = []
    


    def reset(self):
        # env
        self.sConfigs = self.R.randomizeScenarioSpecs().updateDT_careful(15)
        self.satTruth = self.R.randomizeSatTruth(self.inputTleList, self.sConfigs)
        self.stateCatalog = StateCatalog(self.satTruth)
        self.t = self.sConfigs.scenarioEpoch

        self.sensorMap = list2map(self.inputSensorList)
        for sensorKey in self.sensorMap:
            self.sensorMap[sensorKey].reset()
        
        
        # debug
        self.debug_ec = EventCounter()
        self.debug_uniqueManeuverDetections = []

        return self.t, [], self.stateCatalog, False

    def step(self, actions):
        # move to next time
        self.t += self.sConfigs.timeDelta

        # 2. everything needs to move forward to the current time 
        # --> compute truth (and any maneuvers if necessary)
        for satKey in self.satTruth:
            self.satTruth[satKey].tick(self.t)
        
        # 1. Handle Actions
        # i. send new tasks to appropriate sensors (delays)
        for agentKey in actions:
            for sat in actions[agentKey]:
                if actions[agentKey][sat]: # if not do nothing (i.e. False)
                    self.sensorMap[actions[agentKey][sat]].sendSensorTask(self.t, agentKey, sat, self.stateCatalog.currentCatalog[sat])
                    self.debug_ec.increment("tasks sent")
        # ii. execute or continue executing tasks that have arrived to sensor
        for sensor in self.sensorMap:
            self.sensorMap[sensor].tick(self.t, self.satTruth)
        
        
        # 3. gather information
        # i. check to see if any sensor events are available to agent 
        events = []
        for sensor in self.sensorMap:
            events += self.sensorMap[sensor].checkForUpdates(self.t)

        # ii. conduct catalog state updates 
        curratedEvents = [] # events we want to send to agent as a state
        for event in events:
            self.debug_ec.increment(event.type)
            #print(event.satID + "-->"+ str(event.type))
            if event.type == SensorResponse.COMPLETED_NOMINAL:
                # --> update catalog 
                self.stateCatalog.updateState(event.satID, event.newState)
                curratedEvents.append(event)
            elif event.type == SensorResponse.COMPLETED_MANEUVER:
                # --> update catalog with maneuver
                if not self.stateCatalog.wasManeuverAlreadyDetected(self.t, event.satID, event.newState): 
                    self.debug_ec.increment("unique maneuver detection")
                    self.debug_uniqueManeuverDetections.append(event)  
                else:
                    # if not unique, just credit as state update
                    event.type = SensorResponse.COMPLETED_NOMINAL
                curratedEvents.append(event)
                self.stateCatalog.updateState(event.satID, event.newState)
            elif event.type == SensorResponse.DROPPED_LOST:
                # --> lost
                if not self.stateCatalog.wasManeuverAlreadyDetected(self.t, event.satID, event.crystalBallState): 
                    curratedEvents.append(event)
                else:
                    print("object saved just in time")
            
            else: 
                curratedEvents.append(event)

        

        return self.t, curratedEvents, self.stateCatalog, self.t > self.sConfigs.scenarioEnd
    
