from .SensorModality import SensorModality
from skyfield.api import wgs84, load

solar_ephem = load('de421.bsp')

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
        alt, az, distance = (X-self.groundObserver.at(t)).altaz()
        return alt.degrees > 5

    def isVisible(self, X, t):
        alt, az, distance = (X-self.groundObserver.at(t)).altaz()
        if alt.degrees > 10:
            #if self.modality == SensorModality.OPTICS:
            #    return X.is_sunlit(solar_ephem)

            return True

        return False




