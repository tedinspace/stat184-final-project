from skyfield.api import load, N, W
from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import Sensor, SensorModality
import matplotlib.pyplot as plt

eph = load('de421.bsp')
sun = eph['sun']-eph['earth']

t0 = load.timescale().utc(2024, 11, 26, 19, 45, 28) 
t1 = t0 + 1.3

sConfigs = ScenarioConfigs(t0, 24.5)
sConfigs.updateDT_careful(60*5)

bostonOptical = Sensor("boston optical", [42.3583*N, 71.060*W, 0]) 
bostonOptical.updateModality(SensorModality.OPTICS)


cTime = sConfigs.scenarioEpoch

time = []
solarHorizon = []
while cTime < sConfigs.scenarioEnd:
    alt, az, distance = (sun.at(cTime)-bostonOptical.groundObserver.at(cTime)).altaz()
    solarHorizon.append( alt.degrees)
    time.append(cTime.tt)
    cTime += sConfigs.timeDelta

plt.figure(figsize=(10, 6))

plt.plot(time, solarHorizon, color='g', marker='o')
plt.xlabel('Time [UTC]', fontsize=12)
plt.ylabel('Angle of Sun above or below horizon [deg]', fontsize=12)
plt.title('Position of Sun relative to Sensor Test', fontsize=14)
plt.axvline(x=2460641.430844907, color='g', linestyle='--', label='astro twilight start [truth time]')
plt.axvline(x=2460641.453761574, color='r', linestyle='--', label='astro twilight stop [truth time]')
plt.axhline(y=-18, color='r', linestyle='--', label='-18 [deg]/end of atro twilight')  # y=5, red dashed line
plt.axhline(y=-12, color='g', linestyle='--', label='-12 [deg]/start of atro twilight')  # y=5, red dashed line
plt.legend()
plt.show()