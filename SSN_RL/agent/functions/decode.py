

def decodeActions(actions, satKeys, sensorKeys):
    decisions = {}
    for i, satKey in enumerate(satKeys):
        a = actions[i]
        if a == -1:
            decisions[satKey] = False
        else:
            decisions[satKey] = sensorKeys[a]
    
    return decisions

