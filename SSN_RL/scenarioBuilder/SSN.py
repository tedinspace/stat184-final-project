from SSN_RL.environment.Sensor import Sensor
from SSN_RL.environment.SensorModality import SensorModality


MHR = Sensor("mhr", [42.3583, 71.060, 100])
MHR.updateModality(SensorModality.RADAR)

ASCENSION = Sensor("ascension", [-7.678805483927795, -13.265374101627982, 0])
ASCENSION.updateModality(SensorModality.RADAR)

MAUI = Sensor("GEODSS Maui", [20.708259458876853, -156.2567944944128, 0])
MAUI.updateModality(SensorModality.OPTICS)