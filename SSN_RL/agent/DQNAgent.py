import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque

from SSN_RL.agent.functions.decode import decodeActions
import numpy as np
import random 
from SSN_RL.utils.time import MPD


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQNAgent():
    '''DQN Agent
    
    Args
        agentID (str)                 - A unique identifier for the agent.
        assigned_sats (list)          -  List of satellites assigned to the agent.
        assigned_sensors (list)       - List of sensors assigned to the agent.
    (Optional)    
        LR (float, optional)          - Learning rate for the optimizer. Default is 1e-3.
        mem_size (int, optional)      - Size of the replay memory buffer. Default is 1000000.
        batch_size (int, optional)    - Batch size used for training. Default is 50.
        gamma (float, optional)       - Discount factor for future rewards. Default is 0.99.
        epsilon (float, optional)     - Initial epsilon value for the epsilon-greedy strategy. Default is 1.
        epsilon_dec (float, optional) -  The rate at which epsilon decays after each step. Default is 1e-3.
        epsilon_min (float, optional) -  The minimum epsilon value. Default is 0.05.
    
    Sources
        - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
        - https://github.com/FranciscoHu17/BipedalWalker/tree/main 


    '''
    def __init__(self,agentID, assigned_sats, assigned_sensors, LR = 1e-3, mem_size=1000000, batch_size = 50, gamma = 0.99, epsilon=1, epsilon_dec=.9999, epsilon_min=0.05):
        self.agentID = agentID
        self.num_sats = len(assigned_sats)
        self.num_sensors = len(assigned_sensors)
        self.assigned_sats = assigned_sats
        self.assigned_sensors = assigned_sensors
        self.sat2idx = {sat: idx for idx, sat in enumerate(assigned_sats)}

        # hyper-paramters
        self.LR = LR # learning rate; HYPERPARM
        self.gamma = gamma # HP
        self.epsilon = epsilon
        self.eps_threshold = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # memory
        self.memory = ReplayMemory(mem_size, batch_size)
        self.steps_taken = 0

        self.last_tasked = np.ones(self.num_sats)*1e8
        
        # model
        self.model = QNetwork(self.num_sats*2,  self.num_sats, -1, self.num_sensors-1).to(DEVICE)
        self.target_model = QNetwork(self.num_sats*2,self.num_sats, -1, self.num_sensors-1).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.action_space = gym.spaces.Box(low=-1, high=self.num_sensors-1, shape=(self.num_sats,), dtype=np.int32)


    def reset(self):
        self.last_tasked =  np.ones(self.num_sats)*1e8
        
    def step(self, state, action, reward, new_state, done):
        '''call each step of the experiment; handles memory and learning'''
        # update memory
        self.memory.store_transition(state, action, reward, new_state, done)

        # learn after enough experiences
        if len(self.memory) > self.batch_size:
        #if self.steps_taken % 10 == 0 and len(self.memory) > self.batch_size:
            self.learn()
        # update step count
        self.steps_taken += 1

    def getLastSeenLastTasked(self,t, stateCat):
        # compute last tasked by agent in mins
        last_tasked_mins_ago = (t.tt - self.last_tasked)*MPD
        last_tasked_mins_ago[last_tasked_mins_ago < 0] = -1
        # last seen in mins
        lastSeen = np.array([
            stateCat.lastSeen_mins(t, sat) 
            for sat in self.sat2idx.keys()
        ])
        return lastSeen, last_tasked_mins_ago
    
    def encodeState(self, t,stateCat):
        lastSeen, last_tasked_mins_ago = self.getLastSeenLastTasked(t, stateCat)
        return np.concatenate((lastSeen, last_tasked_mins_ago))


    def decide(self,t, events, stateCat):
        # update epsilon
        self.eps_threshold = max(self.epsilon_min, self.eps_threshold * self.epsilon_dec)
        
        lastSeen, last_tasked_mins_ago = self.getLastSeenLastTasked(t, stateCat)

        if np.random.rand() < self.eps_threshold:
            # - random
            #actions = torch.from_numpy(self.action_space.sample())
            bool_arr = ((last_tasked_mins_ago > 30) | (last_tasked_mins_ago == -1)) & (lastSeen > 45)
            actions = np.ones(self.num_sats)*-1
            actions[bool_arr] = np.random.randint(0, self.num_sensors, size=np.sum(bool_arr))
            action_spec = actions
            actions = torch.from_numpy(actions)

        else:
            actions = self.decide_on_policy(np.concatenate((lastSeen, last_tasked_mins_ago)))
            action_spec =  np.round(actions.numpy()).astype(int)
        

        


        # update task records 
        self.last_tasked[ actions != -1] = np.ones(len(self.last_tasked[ actions != -1]))*t.tt
        # return encoded and decoded actions
        return actions, {self.agentID: decodeActions(action_spec, self.assigned_sats, self.assigned_sensors)}


    def decide_on_policy(self, state):
        '''handles agents decision; epsilon greedy'''
        # select action with highest q-value
        state = torch.FloatTensor(np.array(state).reshape(1,-1)).to(DEVICE)
            
        # action that maximizes Q*(s',a';THETA)  
        with torch.no_grad():
            return self.model(state).flatten().cpu().data
            
    def learn(self):
        ''''''
        # Sample random minibatch of transitions from Experience Replay
        state, _, reward, new_state, done = self.memory.sample()
        # Computes Q(s_{curr},a') then chooses columns of actions that were taken for each batch
        q_eval = self.model(state)

        # Clone the model and use it to generate Q learning targets for the main model
        # Also predicts the max Q value for the next state
        q_next = self.target_model(new_state)

        # Q learning targets = r if next state is terminal or
        # Q learning targets = r + GAMMA*(Q(s_{next},a')) if next state is not terminal
        q_target = reward + self.gamma*(q_next) *(1-done)

        # Compute MSE loss
        loss = F.mse_loss(q_eval, q_target)

        # Stochastic gradient descent on the loss function and does backpropragation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,min_action, max_action):
        super(QNetwork, self).__init__()
        self.nn_in = 64
        self.nn_out = 64
        self.l1 = nn.Linear(state_dim, self.nn_in)
        self.l2 = nn.Linear(self.nn_in,  self.nn_out)
        self.l3 = nn.Linear(self.nn_out, action_dim)
        self.max_action = max_action
        self.min_action = min_action
        self.action_range = self.max_action - self.min_action 

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        action_values = torch.tanh(self.l3(x)) * self.action_range / 2 + (self.min_action + self.max_action) / 2
        return action_values

class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    # Add a transition to the memory by basic SARNS convention. 
    def store_transition(self, state, action, reward, new_state, done):
        # If buffer is abuot to overflow, begin rewriting existing memory? 
        self.buffer.append((state, action, reward, new_state, done))

    # Sample only the memory that has been stored. Samples BATCH
    # amount of samples. 
    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.tensor(np.array(states) , dtype=torch.float32).to(DEVICE)
        actions = torch.stack(actions).int().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)


