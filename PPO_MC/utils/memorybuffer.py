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

    def store(self, state, prob, action, reward):
        """Append the sample in the buffer

        Args:
            state (list): state of the agent
            prob (list): prob of selected action or mu for cc
            action (list): performed action
            reward (float): received reward

        Returns:
            None
        """

        self.buffer.append([state, prob, action, reward])

    def sample(self):
        """Get all the samples from the buffer

        Args:
            None

        Returns:
            states (list): states
            probs (list): prob of selected action or mu for cc
            actions (list): performed action
            rewards (float): received reward
        """

        states = np.array([sample[0] for sample in self.buffer])
        probs = np.array([sample[1] for sample in self.buffer])
        actions = np.array([sample[2] for sample in self.buffer])
        rewards = np.array([sample[3] for sample in self.buffer])        

        return states, probs, actions, rewards

    def get_rewards(self, steps):
        """Return the last episode's rewards for the cumulative operations

        Args:
            steps (int): length of the last episode to retrieve rewards

        Returns:
            rewards (list): list of rewards of the last episode
        """

        n_sample = len(self.buffer)
        rewards = [self.buffer[i][3] for i in range(n_sample - steps, n_sample)]
        return rewards    

    def update_rewards(self, rewards, steps):
        """Update the rewards with the cumulative ones in the lat episode

        Args:
            rewards (list): list of cumulated rewards in temporal order
            steps (int): length of the last episode to reinsert rewards

        Returns:
            None
        """

        n_sample = len(self.buffer)
        for i, j in enumerate(range(n_sample - steps,  n_sample)):
            self.buffer[j][3] = rewards[i]

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

