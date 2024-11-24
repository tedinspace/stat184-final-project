from skyfield.api import Topos, load, wgs84, EarthSatellite
from sgp4.api import Satrec, WGS84
from sgp4.ext import rv2coe


import numpy as np

ts = load.timescale()

mu = 3.986004418e5

print(mu)
# MUOS 5: https://www.n2yo.com/satellite/?s=41622
l1 = "1 41622U 16041A   24318.45628365 -.00000122  00000-0  00000+0 0  9999"
l2 = "2 41622   3.5297 296.8182 0199567 257.0196 278.5005  1.00270536 31247"

X = EarthSatellite(l1, l2, 'MUOS 5', ts)
X_man = EarthSatellite(l1, l2, 'MUOS 5', ts)


t = load.timescale().utc(2024, 11, 15, 0, 0, 0)

X1 = X.at(t)


X1.velocity.km_per_s

p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = rv2coe(X1.position.km, X1.velocity.km_per_s, mu)
print("p="+str(p))
print("a="+str(a))
print("ecc="+str(ecc))
print("incl="+str(incl))
print("omega="+str(omega))
print("argp="+str(argp))
print("nu="+str(nu))
print("m="+str(m))
print("arglat="+str(arglat))
print("truelon="+str(truelon))
print("lonper="+str(lonper))

epoch_year = 24
epoch_day = 318.11111111111
T = 2 * np.pi * np.sqrt(a**3 / mu)

# Step 2: Compute mean motion n in revolutions per day
n = 86400 / T  # in rev/day

# Step 3: Convert the parameters into the TLE format
sat_id = 12345  # Example satellite ID (use the actual ID if known)
epoch_str = f"{epoch_year}{int(epoch_day):03d}"  # Format epoch

# Line 1 (with epoch, inclination, eccentricity, argument of perigee, RAAN, etc.)
line1 = (f"1 {sat_id:5d}U {epoch_str} {n:8.4f} {incl*180/np.pi:7.4f} "
            f"{ecc*1e7:7.1f} {omega*180/np.pi:8.4f} {argp*180/np.pi:8.4f} "
            f"{m*180/np.pi:8.4f}    0")

# Line 2 (mean motion, inclination, eccentricity, etc.)
line2 = (f"2 {sat_id:5d} 00000 00000 00000 {n:8.4f} {ecc:7.4f} "
            f"{argp*180/np.pi:8.4f} {argp*180/np.pi:8.4f} {incl*180/np.pi:8.4f} "
            f"00000")


print(line1)
print(l1)
print(line2)