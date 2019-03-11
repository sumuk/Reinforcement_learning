import numpy as np
from collections import deque, namedtuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Qnetwork
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ----------------------
lr = 5e-4  # initial learning rate 
batch_size = 64  # batch size for training
buffer_max_size = int(1e5)  # max size of relay buffer
gamma = 0.99  # discount factor used in calculating total reward
tau = 1e-3  # used for updating a small part of current network to target network
# ----------------------


class Agent():
    '''
    Agent interacts with env and learn the optimal policy by learning optimal value function
    '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.Qlocal = Qnetwork(self.state_size, self.action_size).to(device)  # Local Network
        self.Qtarget = Qnetwork(self.state_size, self.action_size).to(device)  # Taget Network
        self.optim = optim.Adam(self.Qlocal.parameters(), lr)
        self.buffer = replay_buffer(buffer_max_size, batch_size)  # replay buffer
        self.t_step = 0 # used in updating the target network weights from local network

    def learn(self, exp, gamma):
        '''
        takes exp and gamma for training the local network in predicting proper q value
        calculates the next time step q value from next state
        '''
        state, action, reward, next_state, done = exp
        index = self.Qlocal(next_state).detach().max(1)[1].unsqueeze(1)  # double q learning to get max value of action from secondary network 
        q_val = self.Qtarget(next_state).detach().gather(1, index)  # get the q value from the index which gave max value
        y_onehot = torch.zeros(batch_size, self.action_size).cuda()  # update the values for choosen action
        y_onehot.scatter_(1, action, 1)  # creating the one hot vector
        Q_val_n = reward + (gamma * q_val * (1 - done))  # Estimated the Target Q value for state 
        Q_target = y_onehot * Q_val_n  # Traget network Qvalue for given action
        pre = self.Qlocal(state)  # Qvalue estimated by the local network     
        Q_local = y_onehot * pre  # Local network Qvalue for given action
        
        loss = F.mse_loss(Q_local, Q_target) # Loss function
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.update(self.Qlocal, self.Qtarget, tau)  # updating the Target network weight with Local network weight 

    def step(self, state, action, reward, next_state, done):
        '''
        Interacts with the env to get the one step experience and update the replay buffer
        trains Local network one in four times it interacts with env 
        '''
        self.buffer.add(state, action, reward, next_state, done)  # Adding to replay buffer
        self.t_step += 1
        if(self.t_step % 4 == 0):  # Training ones in four times 
            if(len(self.buffer) > batch_size):
                experiences = self.buffer.sample()
                self.learn(experiences, gamma)

    def act(self, state, eps):
        '''
        Given the state provide the appropriate action given by e-greedy policy
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) 
        self.Qlocal.eval()  # network in eval mode to calculate the q value
        with torch.no_grad():
            action_values = self.Qlocal(state)  # Q value estimate for given state  
        self.Qlocal.train()  # network in train mode
        if random.random() > eps:  # e-greedy policy for choosing the action from q values 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update(self, local_model, target_model, tau): 
        '''
        Updates the Target network with some part of local network
        '''
        for l, t in zip(local_model.parameters(), target_model.parameters()):
            t.data.copy_(t.data * (1.0 - tau) + l.data * tau) 


class replay_buffer():
    '''
    Deque holds the state,action,reward,next state and done tuple 
    '''
    def __init__(self, max_size, batch_size):
        '''
        Creates deque of named tuples
        '''
        self.deque = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        '''
        Given state,action,reward,next state and done add's it to replay buffer
        '''
        new_ele = self.experience(state, action, reward, next_state, done)
        self.deque.append(new_ele)

    def sample(self):
        '''
        Randomly samples the replay buffer for state,action,reward,next state and done for training network 
        '''
        exp_sample = random.sample(self.deque, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in exp_sample if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exp_sample if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exp_sample if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exp_sample if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exp_sample if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        provide the number of elements in deque
        '''
        return len(self.deque)
