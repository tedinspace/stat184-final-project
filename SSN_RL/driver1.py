from skyfield.api import Topos, load, wgs84, EarthSatellite
from utils.time import SPD, s2frac, m2frac
from environment.ScenarioConfigs import ScenarioConfigs
from environment.Satellite import Satellite
from environment.Manuever import Maneuver
from scenarioBuilder.clusters import MUOS_CLUSTER
import matplotlib.pyplot as plt


sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 24.5)

muos1 = MUOS_CLUSTER[0]

S1 = Satellite(muos1[0], muos1[1], muos1[2], sConfigs )
S1.reEpoch_init(60)
S1.addManeuvers([Maneuver(1135.1, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])


lat = []
lon = []
alt = []

lat_no = []
lon_no = []
alt_no = []

time = []
cTime = sConfigs.scenarioEpoch
i = 0
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
    i+=1
    time.append(i)
    # 2. translate knowable states to agents
    
    # 3. get agent's responses

    # 4. stage responses impact/rewards/punishments
    
    # iterate
    cTime += sConfigs.timeDelta


plt.figure(figsize=(10, 6))

# Plot latitude, longitude, and altitude
plt.plot(time, lat, label="Latitude (°)", color='b', marker='o')
#plt.plot(time, lon, label="Longitude (°)", color='g', marker='x')
plt.plot(time, lat_no, label="Latitude (°) - no", color='r', marker='o')
#plt.plot(time, lon_no, label="Longitude (°) - no", color='r', marker='x')
#plt.plot(time, alt, label="Altitude (km)", color='r', marker='^')

# Format the plot
plt.xlabel('Time (UTC)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Satellite Position over Time', fontsize=14)
plt.legend()

plt.show()

#epoch = sConfigs.scenarioEpoch
#cadence = 30 / (24 * 3600)  


#time_array = [epoch + i * cadence for i in range(10)]

# Print the times
#for t in time_array:
#    print(t.utc_iso())


#print((time_array[1]-time_array[0])*SPD)

