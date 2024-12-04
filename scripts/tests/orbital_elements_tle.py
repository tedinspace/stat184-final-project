from skyfield.api import load, EarthSatellite
from sgp4.ext import rv2coe
import math
import numpy as np;

G = 6.6743015e-11  # [m^3 kg^-1 s^-2]
M = 5.972e+24 # [kg]
# gravitational parameter
mu = 3.986004418e5 # [km3 s−2]

def computeMeanMotion(a_km):
    return ((G*M)/(a_km*1000)**3)**(1/2)*(60*60*24)/(2*math.pi)

# *    p           - semilatus rectum               km
# *    a           - semimajor axis                 km
# *    ecc         - eccentricity
# *    incl        - inclination                    0.0  to pi rad
# *    omega       - longitude of ascending node    0.0  to 2pi rad
# *    argp        - argument of perigee            0.0  to 2pi rad
# *    nu          - true anomaly                   0.0  to 2pi rad
# *    m           - mean anomaly                   0.0  to 2pi rad
# *    arglat      - argument of latitude      (ci) 0.0  to 2pi rad
# *    truelon     - true longitude            (ce) 0.0  to 2pi rad
# *    lonper      - longitude of periapsis    (ee) 0.0  to 2pi rad

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