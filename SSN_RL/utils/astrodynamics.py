import math


G = 6.6743015e-11  # [m^3 kg^-1 s^-2]
M = 5.972e+24 # [kg]
mu = 3.986004418e5 # [km3 s−2] - gravitational parameter

def computeMeanMotion(a_km):
    '''compute the revolutions per day'''
    return ((G*M)/(a_km*1000)**3)**(1/2)*(60*60*24)/(2*math.pi)


def overrideStr(originalString, value, start, end, pad_char=' '):
    """
    Replaces a part of the string 
    used for creating new TLEs most maneuver
    """
    formatted_value = str(value)
    
    # Ensure it fits within the given range (truncate or pad)
    width = end - start
    if len(formatted_value) > width:
        formatted_value = formatted_value[:width]  # Truncate if too long
    elif len(formatted_value) < width:
        formatted_value = formatted_value.ljust(width, pad_char)  # Pad if too short
    
    # Replace the section of string with the formatted doy
    return originalString[:start] + formatted_value + originalString[end:]


def printTleInfo(l1, l2):
    '''used for debugging/script writing'''
    print('line1')
    print("epoch year: "+l1[18:20])
    print("epoch day.frac: "+l1[20:32])
    print("first direvative of mean motion: "+l1[33:43	])
    print('')
    print('line2')
    print("line #: "+l2[0])
    print("catalog number: "+l2[2:7])
    print("inclination: (deg) "+l2[8:16])
    print("RAAN (deg): "+l2[17:25])
    print("eccentricity: 0."+l2[26:33])
    print("Argument of perigee (degrees): "+l2[34:42])
    print("mean anom (deg): "+l2[43:51])
    print("mean motion: "+l2[52:63])
    print("rev number: "+l2[63:68])
    print("checksum: "+l2[68])