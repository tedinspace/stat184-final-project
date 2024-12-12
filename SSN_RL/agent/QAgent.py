from SSN_RL.agent.functions.decode import decodeActions
import numpy as np
import random
from SSN_RL.utils.time import MPD
from SSN_RL.agent.algorithms.trivial import randomAction
import math

from collections import defaultdict

class QAgent:
    def __init__(self,agentID, assigned_sats, assigned_sensors, epsilon=1, epsilon_dec=.999, epsilon_min=0.05, gamma=.99, alpha=0.01):
        self.agentID = agentID
        self.num_sats = len(assigned_sats)
        self.num_sensors = len(assigned_sensors)
        self.assigned_sats = assigned_sats
        self.assigned_sensors = assigned_sensors
        self.sat2idx = {sat: idx for idx, sat in enumerate(assigned_sats)}

        self.epsilon = epsilon
        self.eps_threshold = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha

        self.qTable = defaultdict( lambda: np.zeros((2, 2)))

        self.last_tasked = np.ones(self.num_sats)*1e8

        self.time_bin_size_mins = 30
        self.max_bins = 5

    def reset(self):
        self.last_tasked =  np.ones(self.num_sats)*1e8


    def getLastSeenLastTasked(self, t, stateCat):
        last_tasked_mins_ago = (t.tt - self.last_tasked) * MPD
        last_tasked_mins_ago[last_tasked_mins_ago < 0] = -1
        lastSeen = np.array([stateCat.lastSeen_mins(t, sat) for sat in self.sat2idx.keys()])
        return lastSeen, last_tasked_mins_ago
    
    def encodeState(self, t,stateCat):
        lastSeen, last_tasked_mins_ago = self.getLastSeenLastTasked(t, stateCat)
        return np.concatenate((lastSeen, last_tasked_mins_ago))
    
    def decide_heuristic(self,t, events, stateCat):

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
        return actions, {self.agentID: decodeActions(actions, self.assigned_sats, self.assigned_sensors)}, tuple([int(a+1) for a in actions])
    
    def decide_on_policy(self,t,events,stateCat):
        state = self.discretizeState(self.encodeState(t, stateCat))
            
        actions = np.unravel_index(np.argmax(self.qTable[state]), self.qTable[state].shape)
        actions_tuple = actions
        actions = np.array(list(actions))-1
        
        
        # update task records 
        self.last_tasked[ actions != -1] = np.ones(len(self.last_tasked[ actions != -1]))*t.tt
        # return encoded and decoded actions
        return actions, {self.agentID: decodeActions(actions, self.assigned_sats, self.assigned_sensors)}, actions_tuple
    
    def decide(self,t, events, stateCat):
        # update epsilon
        self.eps_threshold = max(self.epsilon_min, self.eps_threshold * self.epsilon_dec)

        if random.random() <  self.eps_threshold :
            #actions = randomAction(self.num_sensors, self.num_sats)
            #actions_tuple = tuple(actions+1)
            actions, actions_decoded, actions_tuple = self.decide_heuristic(t, events, stateCat)

        else:
            state = self.discretizeState(self.encodeState(t, stateCat))
            
          
            actions = np.unravel_index(np.argmax(self.qTable[state]), self.qTable[state].shape)
            actions_tuple = actions
            actions = np.array(list(actions))-1
        
        
        # update task records 
        self.last_tasked[ actions != -1] = np.ones(len(self.last_tasked[ actions != -1]))*t.tt
        # return encoded and decoded actions
        return actions, {self.agentID: decodeActions(actions, self.assigned_sats, self.assigned_sensors)}, actions_tuple

    def updateQTable (self, state, action, reward, nextState=None):
        current = self.qTable[state][action]  
        qNext = np.max(self.qTable[nextState]) if nextState is not None else 0
        target = reward + (self.gamma * qNext)
       
        self.qTable[state][action] = current + (self.alpha * (target - current))


    def discretizeState(self,encodedState):
        state = ()
        for s in encodedState:
            if s == -1:
                state += (0,)
            else:

                s = math.ceil(s/self.time_bin_size_mins)
                if s>self.max_bins:
                    s = self.max_bins

                state += (s,)
       
        return state
