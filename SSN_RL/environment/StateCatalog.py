from SSN_RL.utils.time import MPD
class StateCatalog: 
    def __init__(self, initialSatelliteInfo):
        self.previousStateHistory = {}

        self.currentCatalog = {}
        for satKey in initialSatelliteInfo:
            self.currentCatalog[satKey]=StateCatalogEntry(initialSatelliteInfo[satKey].activeObject, initialSatelliteInfo[satKey].activeObject.epoch)
            self.previousStateHistory[satKey] = []

    def updateState(self, satID, newCatalogEntry):
        self.previousStateHistory[satID].append(self.currentCatalog[satID])
        self.currentCatalog[satID] = newCatalogEntry

    def lastSeen(self, t):
        lastSeen = {}
        for satKey in self.currentCatalog:
            lastSeen[satKey]=(t.tt - self.currentCatalog[satKey].stateValidityEpoch.tt )*MPD
        return lastSeen
    def lastSeen_mins(self,t,satKey):
        return (t.tt - self.currentCatalog[satKey].stateValidityEpoch.tt )*MPD
    
    def wasManeuverAlreadyDetected(self, t, satID, newState):
        return sum(self.currentCatalog[satID].activeObject.at(t).position.km-newState.activeObject.at(t).position.km)==0

class StateCatalogEntry: 
    def __init__(self, activeObject, lastSeenEpoch):
        self.activeObject = activeObject
        self.stateValidityEpoch = lastSeenEpoch

