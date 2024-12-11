import numpy as np
from SSN_RL.agent.algorithms.trivial import randomAction, noAction
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.utils.time import MPD
class HeuristicAgent:
    def __init__(self,agentID, assigned_sats, assigned_sensors):
        self.agentID = agentID
        self.num_sats = len(assigned_sats)
        self.num_sensors = len(assigned_sensors)
        self.assigned_sats = assigned_sats
        self.assigned_sensors = assigned_sensors
        self.sat2idx = {sat: idx for idx, sat in enumerate(assigned_sats)}
        self.last_tasked = np.ones(self.num_sats)*1e8
        


    def reset(self):
        self.last_tasked =  np.ones(self.num_sats)*1e8

    def decide(self,t, events, stateCat):

        # compute last seen and last tasked
        lastSeen = np.array([
            stateCat.lastSeen_mins(t, sat) for sat in self.sat2idx.keys()
        ])
        last_tasked_mins_ago = (t.tt - self.last_tasked)*MPD
        last_tasked_mins_ago[last_tasked_mins_ago < 0] = -1
        
        # select actions
        bool_arr = ((last_tasked_mins_ago > 30) | (last_tasked_mins_ago == -1)) & (lastSeen > 60)
        actions = np.ones(self.num_sats)*-1
        actions[bool_arr] = np.random.randint(0, self.num_sensors, size=np.sum(bool_arr))

        # update last tasked
        self.last_tasked[ actions != -1] = np.ones(len(self.last_tasked[ actions != -1]))*t.tt
        
        #actions = randomAction(self.num_sensors, self.num_sats)
        # decode for env
        return decodeActions(actions, self.assigned_sats, self.assigned_sensors)



