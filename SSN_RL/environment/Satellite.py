from skyfield.api import EarthSatellite
from sgp4.ext import rv2coe
import math
import random
import numpy as np
from SSN_RL.utils.astrodynamics import mu, overrideStr, computeMeanMotion
from SSN_RL.utils.time import m2frac, t2doy, h2frac


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
        self.nManeuvers = 0
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
        self.nManeuvers = len(self.maneuverList)

    def reestimateTrueState(self, maneuver, t):
        # > save previous active state for debugging
        self.activeStateRecord.append(self.activeObject)
        # > get pre-maneuver pos at instant of maneuver
        X_at_man = self.activeObject.at(maneuver.time)
        # > get orbital elemnts post-maneuver
        p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = rv2coe(X_at_man.position.km, X_at_man.velocity.km_per_s+maneuver.maneuver/1000, mu)
        # > override line 1
        l1 = self.l1 
        year, doy, fraction_of_day = t2doy(maneuver.time)
        doy = doy+fraction_of_day
        # - year
        l1 = overrideStr(l1, int(str(year)[2:4]), 18, 20)
        l1 = overrideStr(l1, doy, 20, 32)
        # > over ride line2
        l2 = self.l2
        # - inclination
        l2 = overrideStr(l2, math.degrees(incl), 8, 16)
        # - RAAN 
        l2 = overrideStr(l2, math.degrees(omega), 17, 25)
        # - eccentricity 
        l2 = overrideStr(l2, str(ecc).replace('0.',''), 26, 33)
        # - argument of perigee (degrees)
        l2 = overrideStr(l2, math.degrees(argp),34, 42)
        # - mean anom 
        l2 = overrideStr(l2, math.degrees(m),43, 51)
        # - mean motion
        l2 = overrideStr(l2, computeMeanMotion(a),52, 63)

        self.activeObject = EarthSatellite(l1, l2, self.name, self.timeScale)

        #X = self.activeObject.at(t)
        #maneuver.stop



    def tick(self, t):
        for m in self.maneuverList:
            # check for maneuvers
            if m.time < t and not m.occurred:
                self.reestimateTrueState(m, t)
                m.occurred = True
                #print(self.name+" maneuvered at "+str(t.tt)) # for debugging
                
        # change active object if necessary
        return self.activeObject.at(t)
    
    def maneuveredBetween(self,lastEpoch, tNow):
        didManuever = False
        for m in self.maneuverList:
            if m.time > lastEpoch and m.time < tNow:
                return True

        return didManuever

class Maneuver:
    def __init__(self,magDV, hoursIntoScenario, sConfigs):
        #self.dir = np.array([.1, .8, .1]) 
        self.dir = np.array([random.random(), random.random(),random.random()])
        self.dir = self.dir/np.sum(self.dir) # normalize
        self.maneuver = self.dir * magDV
        self.time = sConfigs.scenarioEpoch + h2frac(hoursIntoScenario)
        self.occurred = False
