import numpy as np

def randomAction(nSensors, nSats):
    '''take a random action'''
    return np.random.randint(low=-1, high=nSensors, size=nSats)

def noAction(nSats):
    '''always take no action'''
    return np.ones(nSats)*-1