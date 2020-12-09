"""Memory buffer script

This manages the memory buffer. 
"""

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

    def store(self, state, action, prob, reward, obs_state, done):
        """Append the sample in the buffer

        Args:
            state (list): state of the agent
            action (list): performed action
            prob (float): prob of the performed action
            reward (float): received reward
            obs_state (list): observed state after the action
            done (int): 1 if terminal states in the last episode

        Returns:
            None
        """

        self.buffer.append([state, action, prob, reward, obs_state, done])

    def sample(self):
        """Get the samples from the buffer

        Args:
            None

        Returns:
            states (list): states of the last episode
            actions (list): performed action in the last episode
            probs (float): prob of the performed action in the last episode
            rewards (float): received reward in the last episode
            obs_states (list): observed state after the action in the last episode
            dones (int): 1 if terminal states in the last episode
        """

        states = np.array([sample[0] for sample in self.buffer])
        actions = np.array([sample[1] for sample in self.buffer])
        probs = np.array([sample[2] for sample in self.buffer])
        rewards = np.array([sample[3] for sample in self.buffer])        
        obs_states = np.array([sample[4] for sample in self.buffer])        
        dones = np.array([sample[5] for sample in self.buffer])        
        
        return states, actions, probs, rewards, obs_states, dones

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

