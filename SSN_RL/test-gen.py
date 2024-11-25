from skyfield.api import Topos, load, wgs84, EarthSatellite
from sgp4.api import Satrec, WGS84
from sgp4.ext import rv2coe
import math
import numpy as np;
from utils.astrodynamics import mu, computeMeanMotion, applyManuever
from utils.time import t2doy
from datetime import datetime


l1 = "1 41622U 16041A   24318.45628365 -.00000122  00000-0  00000+0 0  9999"
l2 = "2 41622   3.5297 296.8182 0199567 257.0196 278.5005  1.00270536 31247"
ts = load.timescale()
S = EarthSatellite(l1, l2, 'MUOS 5', ts)
t = S.epoch
X = S.at(t)

print(t)

man = np.array([0, 0, 0])/1000


p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = rv2coe(X.position.km, X.velocity.km_per_s+man, mu)

print('line2')
print("line #: "+l2[0])
print("catalog number: "+l2[2:7])
print("inclination: (deg) "+l2[8:16])
print("--> "+str(math.degrees(incl)))
print("RAAN (deg): "+l2[17:25])
print("---> "+str(math.degrees(omega)))
print("eccentricity: 0."+l2[26:33])
print("--> "+str(ecc))
print("Argument of perigee (degrees): "+l2[34:42])
print("--> "+str(math.degrees(argp)))
print("mean anom (deg): "+l2[43:51])
print("--> "+ str(math.degrees(m)))
print("mean motion: "+l2[52:63])
print("-->  "+str(computeMeanMotion(a)))
print("rev number: "+l2[63:68])
print("checksum: "+l2[68])


print('line1')
print("epoch year: "+l1[18:20])
print("epoch day.frac: "+l1[20:32])
print("first direvative of mean motion: "+l1[33:43	])


year, doy, fraction_of_day = t2doy(t)
# Print the results
print(f"Year: {year}")
print(f"Day of Year (DOY): {doy}")
print(f"Fraction of Day: {fraction_of_day:.6f}")
