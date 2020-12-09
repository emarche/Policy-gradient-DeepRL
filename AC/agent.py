"""AC agent script

This manages the training phase of the discrete agent with MC updates
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

class ActorCritic:
    """
    Class for the AC agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g., std for the Gaussian)

        Returns:
            None
        """

        self.env = env

        self.actor = DeepNetwork.build(env, params['dnn'], actor=True)
        self.actor_optimizer = Adam()

        self.critic = DeepNetwork.build(env, params['dnn'])
        self.critic_optimizer = Adam()

        self.buffer = Buffer()
       
    def get_action(self, state):
        """Get the action to perform

        Args:
            state (list): agent current state

        Returns:
            action (int): sampled action to perform
        """

        probs = self.actor(np.array([state])).numpy()[0]
        action = np.random.choice(self.env.action_space.n, p=probs)
        return action

    def update(self, gamma, steps):
        """Prepare the samples and the cumulative reward to update the network

        Args:
            gamma (float): discount factor for the cumulative reward
            steps (int): n° steps of the episode for the cumulative reward

        Returns:
            None
        """
        
        states, actions, rewards = self.buffer.sample()

        # Compute the discounted cumulative reward as in MC methods
        n_sample = len(states)
        for i in range(n_sample - 2, n_sample - steps, -1):
            rewards[i] += rewards[i + 1] * gamma
       
        # Normalize the reward values to reduce variance
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards + 1e-7)

        # The updates require shape (n° samples, len(reward))
        rewards = rewards.reshape(-1, 1)

        self.update_discrete(states, actions, rewards)

        # After the update, we clear the buffer
        self.buffer.clear()

    def update_discrete(self, states, actions, rewards):
        """Compute the log prob of the performed action to update the actor network's weights
        ∇θJ(θ) ≈ 1/N * ∑ [(∑ ∇θlogPπθ(a∣s)) * (∑ r - V(s))]
        In the discrete case, the Pπθ is directly given by the softmax.
        The objective function to max is the previously cumulated reward - the state value computed by a critic.
        The critic is updated by minimizing mse ∑ r - V(s). It has to learn to predict the correct state values.

        Args:
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update

        Returns:
            None
        """

        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            # Compute Pπθ(a∣s)
            probs = self.actor(states)
            idxs = np.array([[i, action] for i, action in enumerate(actions)])
            action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

            # Take the logPπθ(a∣s)
            log_probs = tf.math.log(action_probs)

            # Compute V(s)
            states_values = self.critic(states)

            # Compute the actual objective 1/N * (log(Pπθ(a|s)) * (∑ r - V(s))
            actor_objective = tf.math.subtract(rewards, states_values)
            actor_objective = tf.math.multiply(rewards, log_probs)
            actor_objective = -tf.math.reduce_mean(actor_objective)

            # Compute the actor gradient and update the network
            grads = tape_a.gradient(actor_objective, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            # Compute the critic loss as R (return) - V(si,t)
            critic_loss = tf.math.square(rewards - states_values)
            critic_loss = tf.math.reduce_mean(critic_loss)

            # Compute the critic gradient and update the network
            critic_grad = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    def train(self, tracker, n_episodes, verbose, params):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., std for the Gaussian)

        Returns:
            None
        """

        mean_reward = deque(maxlen=100)

        gamma = params['gamma']

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                action = self.get_action(state)
                obs_state, obs_reward, done, _ = self.env.step(action)

                self.buffer.store(state, action, obs_reward)
                                        
                ep_reward += obs_reward
                steps += 1
                if done: break  

                state = obs_state.copy()

            self.update(gamma, steps)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        
