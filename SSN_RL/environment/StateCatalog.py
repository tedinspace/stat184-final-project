from SSN_RL.utils.time import MPD
class StateCatalog: 
    def __init__(self, initialSatelliteInfo):
        self.previousStateHistory = {}

        self.currentCatalog = {}
        for satKey in initialSatelliteInfo:
            self.currentCatalog[satKey]=initialSatelliteInfo[satKey].activeObject
            self.previousStateHistory[satKey] = []

    def updateState(self, satID, newActiveObject):
        self.previousStateHistory[satID] = self.currentCatalog[satID]
        self.currentCatalog[satID] = newActiveObject

    def lastSeen_mins(self, t):
        lastSeen = {}
        for satKey in self.currentCatalog:
            lastSeen[satKey]=(t.tt - self.currentCatalog[satKey].epoch.tt )*MPD
        return lastSeen
    
    def wasManeuverAlreadyDetected(self, t, satID, newState):
        return sum(self.currentCatalog[satID].at(t).position.km-newState.at(t).position.km)==0
