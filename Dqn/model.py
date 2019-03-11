import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnetwork(nn.Module):
    '''
    Used for getting the action value for given state
    '''
    def __init__(self, state_size, action_size):
        '''
        initlize layer to get output similar to action size
        '''
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size 
        self.fc1 = nn.Linear(self.state_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.ad = nn.Linear(64, self.action_size) # advantage estimate for given state and action pair
        self.va = nn.Linear(64, 1) # value estimate for given state
    def __call__(self,x):
        '''
        return the action value given the state
        Input
        x state representation
        Return
        action value
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        va = self.va(x)
        ad = self.ad(x)
        q = va + ad - (torch.mean(ad, 1, keepdim=True)) # a(s,a) = q(s,a) - v(s) 
        return q 
