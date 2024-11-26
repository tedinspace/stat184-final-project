from SSN_RL.utils.time import MPD
class StateCatalog: 
    def __init__(self, initialSatelliteInfo):

        self.currentCatalog = {}
        for satKey in initialSatelliteInfo:
            self.currentCatalog[satKey]=initialSatelliteInfo[satKey].activeObject

    def lastSeen_mins(self, t):
        lastSeen = {}
        for satKey in self.currentCatalog:
            lastSeen[satKey]=(t.tt - self.currentCatalog[satKey].epoch.tt )*MPD
        return lastSeen