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

    def __init__(self, size):
        """Instantiate the buffer as an empty list

        Args:
            size (int): maxsize of the buffer

        Returns:
            None
        """

        self.buffer = deque(maxlen=size)

    def store(self, state, action, reward, obs_state, done):
        """Append the sample in the buffer

        Args:
            state (list): state of the agent
            action (list): performed action
            reward (float): received reward
            obs_state (list): observed state after the action
            done (int): 1 if terminal states in the last episode

        Returns:
            None
        """

        self.buffer.append([state, action, reward, obs_state, done])

    def sample(self, batch_size):
        """Get the samples from the buffer

        Args:
            batch_size (int): size of the batch to sample

        Returns:
            states (list): states of the last episode
            actions (list): performed action in the last episode
            rewards (float): received reward in the last episode
            obs_states (list): observed state after the action in the last episode
            dones (int): 1 if terminal states in the last episode
        """
        samples = random.sample(self.buffer, batch_size)

        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])        
        obs_states = np.array([sample[3] for sample in samples])        
        dones = np.array([sample[4] for sample in samples])        
        
        return states, actions, rewards, obs_states, dones

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

