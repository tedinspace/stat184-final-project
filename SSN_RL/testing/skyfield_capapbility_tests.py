from skyfield.api import Topos, load, wgs84, EarthSatellite
from sgp4.api import Satrec, WGS84

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def loadSatelliteFromString():

    ts = load.timescale()
    line1 = '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082'
    line2 = '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473'
    
    return  EarthSatellite(line1, line2, 'ISS (ZARYA)', ts)

def getSatRec():
    line1 = '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082'
    line2 = '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473'
    
    return Satrec.twoline2rv(line1, line2, WGS84)

def createEarthObserver_wgs():
    # return Topos(latitude_degrees=42.3583, longitude_degrees=71.060) # alternate
    return wgs84.latlon(42.3583, 71.060, 100)

def createEarthObserver_topos():
    return Topos(latitude_degrees=42.3583, longitude_degrees=71.060) # alternate

def sensorSatellitePosDiff():

    t = load.timescale().utc(2024, 11, 15, 0, 0, 0)

    sensor = createEarthObserver_topos()
    satellite = loadSatelliteFromString()

    difference = satellite-sensor

    #print(difference.at(t).distance().km)
    #print(difference.at(t).frame_xyz_and_velocity())
    alt, az, distance = difference.at(t).altaz()
    return alt, az, distance

def loadEarthObject():
    eph = load('de421.bsp')
    return eph['earth']



def useObserveClass_although_slow():
    t = load.timescale().utc(2024, 11, 15, 0, 0, 0)
    earth = loadEarthObject()
    boston = createEarthObserver_wgs()
    satellite = loadSatelliteFromString()

    ssb_boston = earth + boston
    ssb_satellite = earth + satellite
    return ssb_boston.at(t).observe(ssb_satellite).apparent()
     


alt, az, distance = sensorSatellitePosDiff()


t = load.timescale().utc(2024, 11, 15, 0, 0, 0)

boston = createEarthObserver_wgs()
satellite = loadSatelliteFromString()


pos = (satellite - boston).at(t)
az, el, the_range, _, _, range_rate = pos.frame_latlon_and_rates(boston)
print(az)
print(el)

print(the_range.km)
ts = load.timescale()
bluffton = wgs84.latlon(+40.8939, -83.8917)
t0 = ts.utc(2014, 1, 23)
t1 = ts.utc(2014, 1, 24)
t, events = satellite.find_events(bluffton, t0, t1, altitude_degrees=30.0)
event_names = 'rise above 30°', 'culminate', 'set below 30°'
for ti, event in zip(t, events):
    name = event_names[event]
    print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)


eph = load('de421.bsp')
sunlit = satellite.at(t).is_sunlit(eph)

for ti, event, sunlit_flag in zip(t, events, sunlit):
    name = event_names[event]
    state = ('in shadow', 'in sunlight')[sunlit_flag]
    print('{:22} {:15} {}'.format(
        ti.utc_strftime('%Y %b %d %H:%M:%S'), name, state,
    ))