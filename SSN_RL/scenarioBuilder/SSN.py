from skyfield.api import N, W
from SSN_RL.environment.Sensor import Sensor, SensorModality


MHR = Sensor("mhr", [42.61762*N, 71.49038*W, 100]) # 42.61762, -71.49038
MHR.updateModality(SensorModality.RADAR)

ASCENSION = Sensor("ascension", [-7.678805483927795, -13.265374101627982, 0])
ASCENSION.updateModality(SensorModality.RADAR)


# GEODSS - https://mostlymissiledefense.com/2012/08/20/space-surveillance-sensors-geodss-ground-based-electro-optical-deep-space-surveillance-system-august-20-2012/
MAUI = Sensor("GEODSS Maui", [20.708259458876853, -156.2567944944128, 0])
MAUI.updateModality(SensorModality.OPTICS)

SOCORRO = Sensor("GEODSS SOCORRO", [32.82*N, 106.66*W, 0])
SOCORRO.updateModality(SensorModality.OPTICS)