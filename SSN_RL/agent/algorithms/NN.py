import torch
import torch.nn as nn

class QNetwork_Shallow(nn.Module):
    def __init__(self, input_dim, output_dim, output_range):
        super(QNetwork_Shallow, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.output_dim = output_dim
        self.min_action = output_range[0]  
        self.max_action = output_range[1]  
        self.action_range = self.max_action - self.min_action 

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        raw_output = self.fc3(x)
        # Apply tanh to ensure values are between -1 and 1, then scale to [min_action, max_action]
        action_values = torch.tanh(raw_output) * self.action_range / 2 + (self.min_action + self.max_action) / 2
        #print("forward", action_values)
        return action_values