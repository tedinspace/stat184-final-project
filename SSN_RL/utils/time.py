from datetime import datetime
from skyfield.api import load

defaultEpoch = load.timescale().utc(2024, 11, 24, 0, 0, 0)
defaultDeltaT = 30

SPD = 86400
MPD = 1440

def hrsAfterEpoch(epoch, time):
    return (time.tt - epoch.tt)*24

def s2frac(s):
    '''seconds --> fraction of a day'''
    return s / SPD

def m2frac(m):
    '''mins --> fraction of a day'''
    return (m*60)/SPD

def h2frac(h):
    return (h*3600)/SPD

def t2doy(t):
    '''skyfield time to year, doy, frac of day conversion'''
    calendar_date = t.utc_iso()
    year = int(calendar_date[:4])
    month = int(calendar_date[5:7])
    day = int(calendar_date[8:10])
    hour = int(calendar_date[11:13])
    minute = int(calendar_date[14:16])
    second = int(calendar_date[17:19])

    # doy
    start_of_year = datetime(year, 1, 1)
    current_date = datetime(year, month, day, hour, minute, second)
    doy = (current_date - start_of_year).days + 1  # +1 because DOY starts from 1

    # fraction of day
    fraction_of_day = (hour * 3600 + minute * 60 + second) / SPD
    return year, doy, fraction_of_day

