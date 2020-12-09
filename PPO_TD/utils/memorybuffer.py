"""Memory buffer script

This manages the memory buffer. 
"""

import random
from collections import deque

import numpy as np

class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self):
        """Instantiate the buffer as an empty list

        Args:
            None

        Returns:
            None
        """

        self.buffer = deque()

    def store(self, state, prob, action, reward, obs_state, done):
        """Append the sample in the buffer

        Args:
            state (list): state of the agent
            prob (list): prob of selected action or mu for cc
            action (list): performed action
            reward (float): received reward
            obs_state (float): observed state from environment
            done (float): if state is terminale

        Returns:
            None
        """

        self.buffer.append([state, prob, action, reward, obs_state, done])

    def sample(self):
        """Get all the samples from the buffer

        Args:
            None

        Returns:
            states (list): states
            probs (list): prob of selected action or mu for cc
            actions (list): performed action
            rewards (float): received reward
            obs_states (float): observed states from environment
            dones (float): if states are terminal
        """

        states = np.array([sample[0] for sample in self.buffer])
        probs = np.array([sample[1] for sample in self.buffer])
        actions = np.array([sample[2] for sample in self.buffer])
        rewards = np.array([sample[3] for sample in self.buffer])        
        obs_states = np.array([sample[4] for sample in self.buffer])        
        dones = np.array([sample[5] for sample in self.buffer])        

        return states, probs, actions, rewards, obs_states, dones

    def clear(self):
        """Clear the buffer after an update of the network

        Args:
            None

        Returns:
            None
        """
        
        self.buffer.clear()

    @property
    def size(self):
        """Return the size of the buffer
        """
        return len(self.buffer)

