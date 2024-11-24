SPD = 86400

def s2frac(s):
    '''seconds --> fraction of a day'''
    return s / SPD

def m2frac(m):
    '''mins --> fraction of a day'''
    return (m*60)/SPD

def h2frac(h):
    return (h*3600)/SPD