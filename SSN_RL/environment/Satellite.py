from skyfield.api import EarthSatellite
from utils.time import m2frac


class Satellite: 
    def __init__(self,name, l1, l2, sConfigs):
        # save original state
        self.name = name
        self.l1 = l1
        self.l2 = l2

        # record of initial reepoching
        self.reEpoched = False # has the TLE epoch been overriden? 
        self.nEpoch = []
        
        # usable objects
        self.activeObject = EarthSatellite(l1, l2, name, sConfigs.ts)
        self.activeStateRecord = []
        # no event object 
        self.noEventObject_debugging = EarthSatellite(l1, l2, name, sConfigs.ts)

        self.maneuverList = []
        # for referencing
        self.timeScale =  sConfigs.ts
        self.scenarioEpoch =  sConfigs.scenarioEpoch

    def reEpoch_init(self, timeLastSeenMinsFromScenarioEpoch):
        '''re epoching allows us to change the state's epoch time without changing the TLE string
        - this gives us better control over our satellites in the scenario'''
        self.reEpoched = True
        self.nEpoch = self.scenarioEpoch - m2frac(timeLastSeenMinsFromScenarioEpoch)
        self.activeObject.epoch = self.nEpoch
        self.noEventObject_debugging.epoch = self.nEpoch
        
    def addManeuvers(self, maneuverList):
        # add them but puts them in time order first
        self.maneuverList = sorted(maneuverList, key=lambda x: x.time)

    def reestimateTrueState(self, maneuver, t):
        # > TODO THIS DOESN'T WORK YET
        # > save previous active state for debugging
        self.activeStateRecord.append(self.activeObject)
        print(self.activeObject)
        print(self.activeObject.at(maneuver.time).velocity.m_per_s)
        print(maneuver.maneuver)
        currStateAtManeuverTime = self.activeObject.at(maneuver.time)
        self.activeObject.epoch = maneuver.time
        self.activeObject.position.km = currStateAtManeuverTime.position.km
        print()
        # > add the instantaneous velocity chance to the maneuver
        self.activeObject.at(maneuver.time).velocity.m_per_s += maneuver.maneuver



    def tick(self, t):
        for m in self.maneuverList:
            # check for maneuvers
            if m.time < t and not m.occurred:
                self.reestimateTrueState(m, t)
                m.occurred = True
                
        # change active object if necessary
        return self.activeObject.at(t)
