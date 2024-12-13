import numpy as np
from SSN_RL.agent.functions.decode import decodeActions
from SSN_RL.utils.time import MPD
from SSN_RL.environment.Sensor import SensorResponse

class LinearQAgent:
    def __init__(self, agentID, assigned_sats, assigned_sensors, 
                 learning_rate=0.01, gamma=0.99, epsilon=1.0, 
                 epsilon_dec=0.999, epsilon_min=0.05):
        self.agentID = agentID
        self.num_sats = len(assigned_sats)
        self.num_sensors = len(assigned_sensors)
        self.assigned_sats = assigned_sats
        self.assigned_sensors = assigned_sensors
        self.sat2idx = {sat: idx for idx, sat in enumerate(assigned_sats)}
        
        # Learning parameters
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_threshold = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        
        # State tracking
        self.last_tasked = np.ones(self.num_sats) * 1e8
        self.last_state = None
        self.last_action = None
        
        # Initialize weights: [num_actions, num_features]
        # Each action (including no-op) needs its own set of weights
        self.num_features = self._get_feature_size()
        self.num_actions = self.num_sensors + 1  # Include no-op action
        self.weights = np.zeros((self.num_sats, self.num_actions, self.num_features))
        
    def _get_feature_size(self):
        """Calculate feature vector size per satellite"""
        # Features per satellite:
        # 1. Normalized last seen time
        # 2. Normalized last tasked time
        return 2
    
    def _extract_features(self, state):
        """Convert raw state into feature vector for each satellite"""
        lastSeen = state[:self.num_sats]
        lastTasked = state[self.num_sats:]
        
        features = np.zeros((self.num_sats, self.num_features))
        for sat_idx in range(self.num_sats):
            # Normalize times using sigmoid
            norm_lastSeen = 2.0 / (1.0 + np.exp(-lastSeen[sat_idx]/240.0)) - 1.0
            norm_lastTasked = 2.0 / (1.0 + np.exp(-lastTasked[sat_idx]/30.0)) - 1.0
            
            features[sat_idx] = [norm_lastSeen, norm_lastTasked]
            
        return features
    
    def _get_q_value(self, features, sat_idx, action):
        """Calculate Q-value for given features and action for a specific satellite"""
        return np.dot(features[sat_idx], self.weights[sat_idx, action + 1])
    
    def _update_weights(self, features, actions, reward, next_features):
        """Update weights for each satellite-action pair"""
        for sat_idx in range(self.num_sats):
            action = actions[sat_idx]
            current_q = self._get_q_value(features, sat_idx, action)
            
            # Calculate max Q-value for next state
            next_q_values = [self._get_q_value(next_features, sat_idx, a-1) 
                           for a in range(self.num_actions)]
            max_next_q = max(next_q_values)
            
            # Q-learning update
            target = reward + self.gamma * max_next_q
            td_error = target - current_q
            
            # Update weights for this satellite-action pair
            self.weights[sat_idx, action + 1] += self.alpha * td_error * features[sat_idx]
    
    def reset(self):
        self.last_tasked = np.ones(self.num_sats) * 1e8
        self.last_state = None
        self.last_action = None
    
    def getLastSeenLastTasked(self, t, stateCat):
        last_tasked_mins_ago = (t.tt - self.last_tasked) * MPD
        last_tasked_mins_ago[last_tasked_mins_ago < 0] = -1
        
        lastSeen = np.array([
            stateCat.lastSeen_mins(t, sat) 
            for sat in self.sat2idx.keys()
        ])
        return lastSeen, last_tasked_mins_ago
    
    def encodeState(self, t, stateCat):
        lastSeen, last_tasked_mins_ago = self.getLastSeenLastTasked(t, stateCat)
        return np.concatenate((lastSeen, last_tasked_mins_ago))
    
    def decide(self, t, events, stateCat):
        # Update epsilon
        self.eps_threshold = max(self.epsilon_min, 
                               self.eps_threshold * self.epsilon_dec)
        
        # Get current state and features
        current_state = self.encodeState(t, stateCat)
        features = self._extract_features(current_state)
        
        # Choose actions for each satellite
        actions = np.zeros(self.num_sats, dtype=int)
        
        if np.random.random() < self.eps_threshold:
            # Exploration: Use heuristic-based random actions
            bool_arr = ((current_state[self.num_sats:] > 15) | 
                       (current_state[self.num_sats:] == -1)) & (current_state[:self.num_sats] > 30)
            actions = np.ones(self.num_sats, dtype=int) * -1
            actions[bool_arr] = np.random.randint(0, self.num_sensors, size=np.sum(bool_arr))
        else:
            # Exploitation: Choose best actions based on Q-values
            for sat_idx in range(self.num_sats):
                q_values = [self._get_q_value(features, sat_idx, a-1) 
                          for a in range(self.num_actions)]
                actions[sat_idx] = np.argmax(q_values) - 1
        
        # Update last tasked times
        self.last_tasked[actions != -1] = t.tt
        
        # Learning update if we have previous state-action pair
        if self.last_state is not None:
            reward = self._calculate_reward(events)
            last_features = self._extract_features(self.last_state)
            self._update_weights(last_features, self.last_action, 
                               reward, features)
        
        self.last_state = current_state.copy()
        self.last_action = actions.copy()
        
        # Return actions and decoded format
        return actions, {self.agentID: decodeActions(actions, 
                                                   self.assigned_sats, 
                                                   self.assigned_sensors)}
    
    def _calculate_reward(self, events):
        reward = 0
        for event in events:
            if event.agentID == self.agentID:
                if event.type == SensorResponse.INVALID:
                    reward -= 50
                elif event.type == SensorResponse.INVALID_TIME:
                    reward -= 2.5
                elif event.type == SensorResponse.DROPPED_SCHEDULING:
                    reward -= 2
                elif event.type == SensorResponse.DROPPED_LOST:
                    reward -= 100
                elif event.type == SensorResponse.COMPLETED_NOMINAL:
                    reward += 1
                elif event.type == SensorResponse.COMPLETED_MANEUVER:
                    reward += 50
        return reward
