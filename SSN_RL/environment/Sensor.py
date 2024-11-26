from SSN_RL.environment.SensorModality import SensorModality
from skyfield.api import wgs84, load

eph = load('de421.bsp')
sun = eph['sun']-eph['earth']

class Sensor:
    def __init__(self, name, lla):
        self.name = name
        self.lat = lla[0]
        self.lon = lla[1]
        self.alt = lla[2]
        self.modality = SensorModality.UNSPEC
        self.groundObserver = wgs84.latlon(self.lat, self.lon, self.alt)


    def updateModality(self, modeEnum):
        self.modality = modeEnum

    def hasLineOfSight(self, X, t):
        alt, _, _ = (X-self.groundObserver.at(t)).altaz()
        return alt.degrees > 5

    def isVisible(self, X, t):
        alt, _, _ = (X-self.groundObserver.at(t)).altaz()
        if alt.degrees > 10:
            if self.modality == SensorModality.OPTICS:
                alt_solar, _, _ = (sun.at(t)-self.groundObserver.at(t)).altaz()
                return alt_solar.degrees < -12 # up until nautical twilight starts
            return True

        return False




