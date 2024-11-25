import math
from sgp4.ext import rv2coe
from datetime import datetime


G = 6.6743015e-11  # [m^3 kg^-1 s^-2]
M = 5.972e+24 # [kg]
# gravitational parameter
mu = 3.986004418e5 # [km3 s−2]

def computeMeanMotion(a_km):
    return ((G*M)/(a_km*1000)**3)**(1/2)*(60*60*24)/(2*math.pi)


def overrideStr(originalString, value, start, end, pad_char=' '):
    """
    Replaces a part of the string 
    """
    # Convert doy to string with specified precision
    #formatted_value = f"{value:.{precision}f}"
    formatted_value = str(value)
    
    # Ensure it fits within the given range (truncate or pad)
    width = end - start
    if len(formatted_value) > width:
        formatted_value = formatted_value[:width]  # Truncate if too long
    elif len(formatted_value) < width:
        formatted_value = formatted_value.ljust(width, pad_char)  # Pad if too short
    
    # Replace the section of l1 with the formatted doy
    return originalString[:start] + formatted_value + originalString[end:]