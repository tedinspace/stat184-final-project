from skyfield.api import Topos, load, wgs84, EarthSatellite
from sgp4.api import Satrec, WGS84
import numpy as np

ts = load.timescale()


# MUOS 5: https://www.n2yo.com/satellite/?s=41622
l1 = "1 41622U 16041A   24318.45628365 -.00000122  00000-0  00000+0 0  9999"
l2 = "2 41622   3.5297 296.8182 0199567 257.0196 278.5005  1.00270536 31247"

X = EarthSatellite(l1, l2, 'MUOS 5', ts)
X_man = EarthSatellite(l1, l2, 'MUOS 5', ts)


t = load.timescale().utc(2024, 11, 15, 0, 0, 0)

X1 = X.at(t)

print(X1.position.km)
print(X1.velocity.m_per_s)

#imes = ts.utc(2024, 11, 15, np.arange(0, 24, 0.5))  # Every 30 mins
modified_position = X1.position.km + np.array([100, 0, 0])  # Shift by 100 km along x-axis
modified_velocity = X1.velocity.m_per_s + np.array([10, 0, 0])  # Increase by 10 m/s along x-axis

X1.position.km = modified_position
print(X1.position.km )
