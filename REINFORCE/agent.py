"""REINFORCE agent script with baseline

This manages the training phase of both the discrete and the continuous agents
"""

from collections import deque

import yaml
import numpy as np

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(seed)

from utils.deepnetwork import DeepNetwork
from utils.memorybuffer import Buffer

class Reinforce:
    """
    Class for the Reinforce agent
    """

    def __init__(self, env, params, continuous):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g., std for the Gaussian)
            continuous (bool): wheter is continuous control (cc)

        Returns:
            None
        """

        self.env = env
        self.continuous = continuous
        if continuous: self.std = params['std']

        self.model = DeepNetwork.build(env, params['dnn'], continuous)
        self.optimizer = Adam()

        self.buffer = Buffer()
       
    def get_action(self, state, std=1.0):
        """Get the action to perform

        Args:
            state (list): agent current state
            std (float): std for the Gaussian in cc

        Returns:
            action (list): sampled action to perform, integers or floats
            mu (float): Gaussian's mean in cc
        """

        if self.continuous:
            mu = self.model(np.array([state])).numpy()[0]
            action = np.random.normal(loc=mu, scale=std**2)    
            return action, mu

        probs = self.model(np.array([state])).numpy()[0]
        action = np.random.choice(self.env.action_space.n, p=probs)
        return action

    def update(self, gamma, steps, std=1.0, use_baseline=False):
        """Prepare the samples and the cumulative reward to update the network

        Args:
            gamma (float): discount factor for the cumulative reward
            steps (int): n° steps of the episode for the cumulative reward
            std (float): std for the Gaussian in cc
            use_baseline (bool): the baseline improves the variance issue of Reinforce

        Returns:
            None
        """

        states, actions, rewards = self.buffer.sample()

        # Compute the discounted cumulative reward as in MC methods
        n_sample = len(states)
        for i in range(n_sample - 2, n_sample - steps, -1):
            rewards[i] += rewards[i + 1] * gamma
       
        # We subtract the mean episode reward as baseline to reduce variance
        if use_baseline:
            baseline = np.mean(rewards)   
            rewards -= baseline

        # Normalize the reward values to reduce variance
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards + 1e-7)

        # The updates require shape (n° samples, len(reward))
        rewards = rewards.reshape(-1, 1)

        if self.continuous:
            self.update_continuous(states, actions, rewards, std)
        else:
            self.update_discrete(states, actions, rewards)

        # After the update, we clear the buffer
        self.buffer.clear()

    def update_discrete(self, states, actions, rewards):
        """Compute the log prob of the performed action to update the network's weights
        ∇θJ(θ) ≈ 1/N * ∑ [(∑ ∇θlogPπθ(a∣s)) * ((∑ r) - baseline)]
        In the discrete case, the Pπθ is directly given by the softmax.
        The objective function to max is the previously cumulated reward, optionally - baseline

        Args:
            states (list): episode's states for the update
            actions (int): episode's actions for the update
            rewards (float): episode's rewards for the update

        Returns:
            None
        """

        with tf.GradientTape() as tape:
            # Compute Pπθ(a|s)
            probs = self.model(states)
            idxs = np.array([[i, action] for i, action in enumerate(actions)])
            action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

            # Take the log(Pπθ(a|s))
            log_probs = tf.math.log(action_probs)

            # Compute the actual objective 1/N * (log(Pπθ(a|s)) * (∑ r - baseline)
            objective = tf.math.multiply(rewards, log_probs)
            # We negate it as we want to max the objective
            objective = -tf.math.reduce_mean(objective)

            # Compute the gradient and update the network
            grads = tape.gradient(objective, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_continuous(self, states, actions, rewards, std):
        """Compute the log prob of the performed action to update the network's weights
        ∇θJ(θ) ≈  1/N * ∑ [(∑ ∇θlogPπθ(a∣s)) * ((∑ r) - baseline)]
        In the continuous case, the Pπθ is computed using a Gaussian distribution, where the mu is computed by the network.
        The objective function to max is the previously cumulated reward.

        Args:
            states (list): episode's states for the update
            actions (int): episode's actions for the update
            rewards (float): episode's rewards for the update
            std (float): std deviation of the Gaussian

        Returns:
            None
        """

        with tf.GradientTape() as tape:

            # Compute the Gaussian's mu from the network, then Pπθ(ai,t∣si,t)
            mu = self.model(states)
            gauss_num = tf.math.exp(-0.5 * ((actions - mu) / (std))**2)
            gauss_denom = std * tf.sqrt(2 * np.pi)
            gauss_probs = gauss_num / gauss_denom

            # Sum/average/do nothing with the contribution of the n continuous actions
            gauss_probs = tf.math.reduce_mean(gauss_probs, axis=1, keepdims=True)  

            # Take the logPπθ(a∣s)
            log_probs = tf.math.log(gauss_probs)

            # Compute the actual objective 1/N * (log(Pπθ(a|s)) * (∑ r)
            objective = tf.math.multiply(rewards, log_probs)
            # We negate it as we want to max the objective
            objective = -tf.math.reduce_mean(objective)

            # Compute the gradient and update the network
            grads = tape.gradient(objective, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, tracker, n_episodes, verbose, params, baseline):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., std for the Gaussian)
            baseline (bool): wether to use the baseline to reduce Reinforce variance

        Returns:
            None
        """
        
        mean_reward = deque(maxlen=100)

        gamma = params['gamma']
        std = params['std'] # continuous agent's std dev

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  

            state = self.env.reset()

            while True:
                if self.continuous: 
                    action, _ = self.get_action(state, std)
                else: 
                    action = self.get_action(state)

                obs_state, obs_reward, done, _ = self.env.step(action)

                self.buffer.store(state, action, obs_reward)

                ep_reward += obs_reward
                steps += 1
                if done: break  

                state = obs_state

            self.update(gamma, steps, std, baseline)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        
