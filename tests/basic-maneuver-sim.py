from skyfield.api import load
from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Satellite import Satellite
from SSN_RL.environment.Manuever import Maneuver
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
import matplotlib.pyplot as plt


sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 24.5)

muos1 = MUOS_CLUSTER[0]

S1 = Satellite(muos1[0], muos1[1], muos1[2], sConfigs )
S1.reEpoch_init(60)
S1.addManeuvers([Maneuver(10, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])


lat = []
lon = []
alt = []

lat_no = []
lon_no = []
alt_no = []

time = []
cTime = sConfigs.scenarioEpoch
while cTime < sConfigs.scenarioEnd:
    # 1. everything needs to move forward to the current time 
    X1 = S1.tick(cTime)

    X1_no = S1.noEventObject_debugging.at(cTime)
    sb_no = X1_no.subpoint()
    lat_no.append(sb_no.latitude.degrees)
    lon_no.append(sb_no.longitude.degrees)
    alt_no.append(sb_no.elevation.km)

    sb = X1.subpoint()
    lat.append(sb.latitude.degrees)
    lon.append(sb.longitude.degrees)
    alt.append(sb.elevation.km)
    time.append(cTime.tt)
    # 2. translate knowable states to agents
    
    # 3. get agent's responses

    # 4. stage responses impact/rewards/punishments
    
    # iterate
    cTime += sConfigs.timeDelta


plt.figure(figsize=(10, 6))

plt.plot(time, lat_no, label="Latitude (°) no maneuver", color='g', marker='o')
plt.plot(time, lon_no, label="Longitude (°) no maneuver", color='g', marker='x')
plt.plot(time, lat, label="Latitude (°) change with maneuver", color='r', marker='o')
plt.plot(time, lon, label="Longitude (°) change with maneuver", color='r', marker='x')

for m in S1.maneuverList:
    plt.axvline(x=m.time.tt, color='r', linestyle='--', label='maneuver')


plt.xlabel('Time (UTC)', fontsize=12)
plt.ylabel('Latitude/Longitude [degree]', fontsize=12)
plt.title('Satellite Position (Lat/Lon) with and without maneuver', fontsize=14)
plt.legend()

plt.show()

plt.plot(time, alt, label="Altitude (km) change with maneuver", color='r', marker='^')
plt.plot(time, alt_no, label="Altitude (km) no maneuver", color='g', marker='^')
for m in S1.maneuverList:
    plt.axvline(x=m.time.tt, color='r', linestyle='--', label='maneuver')
plt.xlabel('Time (UTC)', fontsize=12)
plt.ylabel('Altitude [km]', fontsize=12)
plt.title('Satellite Position (Altitude) with and without maneuver', fontsize=14)
plt.legend()
plt.show()