import numpy as np

def randomAction(nSensors, nSats):
    return np.random.randint(low=-1, high=nSensors, size=nSats)