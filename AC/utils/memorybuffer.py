"""Memory buffer script

This manages the memory buffer. 
Reinforce uses a Monte Carlo estimation of the return, hence it is episodic.
"""

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

        self.buffer = []

    def store(self, state, action, reward):
        """Append the sample in the buffer

        Args:
            state (list): state of the agent
            action (list): performed action
            reward (float): received reward

        Returns:
            None
        """

        self.buffer.append([state, action, reward])

    def sample(self):
        """Get the samples from the buffer

        Args:
            None

        Returns:
            states (list): states of the last episode
            actions (list): performed action in the last episode
            rewards (list): received reward in the last episode
        """

        states = np.array([sample[0] for sample in self.buffer])
        actions = np.array([sample[1] for sample in self.buffer])
        rewards = np.array([sample[2] for sample in self.buffer])        
        return states, actions, rewards

    def clear(self):
        """Clear the buffer after an update of the network

        Args:
            None

        Returns:
            None
        """
        
        self.buffer.clear()

