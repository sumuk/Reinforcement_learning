import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed,replaybuffer):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.steps=0
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*2, action_size*2, random_seed).to(device)
        self.critic_target = Critic(state_size*2, action_size*2, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = replaybuffer 
#         self.actor_target.load_state_dict(self.actor_local.state_dict())
#         self.critic_target.load_state_dict(self.critic_local.state_dict())
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action +=  self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()                   

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
def learn(agents,buffer,agent_no):
    if len(buffer) > BATCH_SIZE :
            experiences = buffer.sample()
            states, actions, rewards, next_states, dones = experiences
            
            next_actions=torch.zeros((256,2,2))
            for i in range(2):
                next_actions[:,i,:]=agents[i].actor_target(next_states[:,i,:].view(256,24))
            q_next = agents[agent_no].critic_target(next_states.view(256,48),next_actions.view(256,4)).squeeze(1)
            q_target = rewards[:,agent_no] + (GAMMA*(1-dones[:,agent_no])*q_next)
            
            q_expected = agents[agent_no].critic_local(states.view(256,48),actions.view(256,4)).squeeze(1)
            
            agents[agent_no].critic_loss = F.mse_loss(input=q_expected, target=q_target)
            agents[agent_no].critic_optimizer.zero_grad()
            agents[agent_no].critic_loss.backward()
            agents[agent_no].critic_optimizer.step()
            
            caction = [agents[i].actor_local(states[:,i,:].view(256,24)) if i == agent_no else agents[i].actor_local(states[:,i,:].view(256,24)).detach() for i in range(2)]
            caction = torch.stack(caction,dim=1)
            agents[agent_no].actor_loss=-agents[agent_no].critic_local(states.view(256,48),caction.view(256,4)).mean()
            agents[agent_no].actor_optimizer.zero_grad()
            agents[agent_no].actor_loss.backward()
            agents[agent_no].actor_optimizer.step()  
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self,*args,**kargs):
        """Add a new experience to memory."""
        data=[np.expand_dims(i,axis=0) for i in args]
        e = self.experience(*data)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)