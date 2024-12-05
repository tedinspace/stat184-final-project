from SSN_RL.environment.Sensor import SensorResponse
import numpy as np

class QTableAgent:
    def __init__(self, agentID, assignedSats, availableSensors, 
                 learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.agentID = agentID
        self.assignedSats = assignedSats
        self.availableSensors = availableSensors
        self.nSensors = len(self.availableSensors)
        self.nSats = len(self.assignedSats)
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize satellite index mapping
        self.sat2idx = {sat: idx for idx, sat in enumerate(self.assignedSats)}
        
        # Track lost satellites
        self.lostSatellites = set()
        self.cumulativeRewardOrCost = 0
        
        # Initialize Q-table: state space x action space
        # State space: 2 event types + lastSeen for each satellite
        # Action space: -1 (do nothing) or select a sensor (0 to nSensors-1)
        self.state_size = 3 * self.nSats  # eventEncoding (2) + lastSeen (1) per satellite
        self.action_size = self.nSensors + 1  # Include -1 as an action
        
        # Initialize Q-table with small random values
        self.Q = {}

    
    def reset(self):
        self.lostSatellites = set()
        self.cumulativeRewardOrCost = 0


    def _get_state_key(self, state):
        """Convert continuous state to discrete key for Q-table"""
        lastSeen = state[self.nSats*2:]
        discretized_lastSeen = np.digitize(lastSeen, bins=[60, 120, 240, 480])
        
        return tuple(np.concatenate([state[:self.nSats*2], discretized_lastSeen]))
    
    def _epsilon_greedy_action(self, state_key, sat_idx):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(low=-1, high=self.nSensors)
        
        state_sat_key = state_key + (sat_idx,)
        if state_sat_key not in self.Q:
            self.Q[state_sat_key] = np.zeros(self.action_size)
        
        return np.argmax(self.Q[state_sat_key]) - 1  # Shift to match -1 to nSensors-1 range

    def decide(self, t, events, stateCatalog):
        current_state, reward = self.encodeCurrentState(t, events, stateCatalog)
        self.cumulativeRewardOrCost += reward
        state_key = self._get_state_key(current_state)
        
        actions = np.array([
            self._epsilon_greedy_action(state_key, i) 
            for i in range(self.nSats)
        ])
        
        if hasattr(self, '_prev_state_key'):
            for i in range(self.nSats):
                prev_state_sat_key = self._prev_state_key + (i,)
                curr_state_sat_key = state_key + (i,)
                
                if prev_state_sat_key not in self.Q:
                    self.Q[prev_state_sat_key] = np.zeros(self.action_size)
                if curr_state_sat_key not in self.Q:
                    self.Q[curr_state_sat_key] = np.zeros(self.action_size)
                
                # Q-learning update
                prev_action = self._prev_actions[i] + 1
                max_next_q = np.max(self.Q[curr_state_sat_key])
                self.Q[prev_state_sat_key][prev_action] += self.learning_rate * (
                    reward + self.discount_factor * max_next_q - 
                    self.Q[prev_state_sat_key][prev_action]
                )
        
        self._prev_state_key = state_key
        self._prev_actions = actions
        
        #actions = np.random.randint(low=-1, high=self.nSensors, size=self.nSats)

        decisions = {}
        for i in range(self.nSats):
            sat = self.assignedSats[i]
            a = actions[i]
            
            if a == -1:
                decisions[sat] = False
            else:
                decisions[sat] = self.availableSensors[a]
        
        return decisions , reward

    def encodeCurrentState(self, t, events, stateCatalog):
        eventEncoding = np.zeros(self.nSats)
        rewardOrCost = 0
        
        for event in events:
            if event.agentID == self.agentID:
                sat_idx = self.sat2idx[event.satID]
                
                if event.type == SensorResponse.INVALID:
                    rewardOrCost -= 50
                elif event.type == SensorResponse.INVALID_TIME:
                    rewardOrCost -= 2.5
                elif event.type == SensorResponse.DROPPED_SCHEDULING:
                    rewardOrCost -= 2
                elif event.type == SensorResponse.DROPPED_LOST:
                    eventEncoding[sat_idx] = 2
                    rewardOrCost -= 100
                    self.lostSatellites.add(event.satID)
                elif event.type == SensorResponse.COMPLETED_NOMINAL:
                    rewardOrCost += 1
                elif event.type == SensorResponse.COMPLETED_MANEUVER:
                    eventEncoding[sat_idx] = 1
                    rewardOrCost += 50
        
        lastSeen = np.array([
            stateCatalog.lastSeen_mins(t, sat) 
            for sat in self.assignedSats
        ])
        
        rewardOrCost -= np.sum(lastSeen > 240) * 2


        #print(self.cumulativeRewardOrCost)
        
        return np.concatenate((eventEncoding, lastSeen)), rewardOrCost


        



