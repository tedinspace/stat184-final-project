from skyfield.api import Topos, load, wgs84, EarthSatellite
from utils.time import SPD, s2frac, m2frac
from environment.ScenarioConfigs import ScenarioConfigs
from environment.Satellite import Satellite


sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 24.5)

# source: https://www.n2yo.com/satellite/?s=38093
l1_muos = '1 38093U 12009A   24328.84934740 -.00000118  00000-0  00000+0 0  9993'
l2_muos = '2 38093   4.3406  45.8931 0049934 358.5963 224.6576  1.00270913 46735'


S1 = Satellite("MUOS 1", l1_muos, l2_muos,sConfigs )
S1.reEpoch_init(60)

cTime = sConfigs.scenarioEpoch




M = EarthSatellite(l1_muos, l2_muos, 'muos', sConfigs.ts)
M.epoch = sConfigs.scenarioEpoch - m2frac(60)
#epoch = sConfigs.scenarioEpoch
#cadence = 30 / (24 * 3600)  


#time_array = [epoch + i * cadence for i in range(10)]

# Print the times
#for t in time_array:
#    print(t.utc_iso())


#print((time_array[1]-time_array[0])*SPD)

