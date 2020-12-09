"""A2C agent script 

This manages the training phase of the on-policy with step by step updates and off-policy version with importance sampling.
"""

import random
from collections import deque

import yaml
import numpy as np

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()
    
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(seed)

from utils.deepnetwork import DeepNetwork
from utils.memorybuffer import Buffer

class AdvantageActorCritic:
    """
    Class for the A2C agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer for the off-policy version

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)

        Returns:
            None
        """

        self.env = env

        self.actor = DeepNetwork.build(env, params['dnn'], actor=True)
        self.actor_optimizer = Adam()

        self.critic = DeepNetwork.build(env, params['dnn'])
        self.critic_optimizer = Adam()

        self.buffer = Buffer(params['buffer'])

    def get_action(self, state):
        """Get the action to perform

        Args:
            state (list): agent current state

        Returns:
            action (int): sampled action to perform
            probs[action] (float): probability of such action
        """

        probs = self.actor(np.array([state])).numpy()[0]
        action = np.random.choice(self.env.action_space.n, p=probs)
        return action, probs[action]

    def update(self, gamma, offpolicy, batch_size):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            offpolicy (bool): wether is off-policy or on-policy A2C
            batch_size (int): batch size for the off-policy A2C

        Returns:
            None
        """

        states, actions, probs, rewards, obs_states, dones = self.buffer.sample()

        if offpolicy:
            batch_size = min(self.buffer.size, batch_size)
            n_batch = int(len(states) / batch_size)
            states = np.array_split(states, n_batch)[0]
            actions = np.array_split(actions, n_batch)[0]
            probs = np.array_split(probs, n_batch)[0]
            rewards = np.array_split(rewards, n_batch)[0]
            obs_states = np.array_split(obs_states, n_batch)[0]
            dones = np.array_split(dones, n_batch)[0]

        # The updates require shape (n° samples, len(metric))
        probs = probs.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
           
        self.update_discrete(gamma, states, actions, probs, rewards, obs_states, dones, offpolicy)

         # After the update, we clear the buffer in the on-policy A2C
        if not offpolicy:
            self.buffer.clear()           

    def update_discrete(self, gamma, states, actions, old_probs, rewards, obs_states, dones, offpolicy):
        """Compute the log prob of the performed action to update the actor network's weights
        ∇θJ(θ) ≈ 1/N * ∑ [(∑ ∇θlogPπθ(a∣s)) * A]. Advantage A = Q(s,a) - V(s), and we use the critic for both the terms as Q(s,a) = r + γ * V(s')
        In the discrete case, the Pπθ is directly given by the softmax.
        The objective function to max is the advantage.
        The critic is updated by minimizing mse Q(s,a) - V(s). It has to learn to predict the advantage.
        Off-policy multiplies the actor objective by the importance sampling πθ(a∣s)/π'θ(a∣s) , where π'θ is the current policy

        Args:
            gamma (float): discount factor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            probs (list): episode's probs of the action for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update
            offpolicy (bool): episode's rewards for the update

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

            # Compute V(s), V(s')
            states_values = self.critic(states)
            osb_states_values = self.critic(obs_states)
 
            # Compute Q(s,a) - V(s) = (r + γ * V(s')) - V(s)
            states_actions_values = rewards + gamma * osb_states_values * dones

            actor_objective = tf.math.subtract(states_actions_values, states_values) 

            # Compute the actual objective 1/N * ∑ [(∑ ∇θlogPπθ(a∣s)) * A]
            actor_objective = tf.math.multiply(actor_objective, log_probs)

            # If off-policy A2C, multiply actor_objective by the importance sampling
            if offpolicy:
                # Compute the importance sampling πθ(a∣s)/π'θ(a∣s) 
                importance_s = tf.math.divide(old_probs, action_probs + 1e-10)

                actor_objective = tf.math.multiply(actor_objective, importance_s)

            actor_objective = -tf.math.reduce_mean(actor_objective)

            # Compute the actor gradient and update the network
            grads = tape_a.gradient(actor_objective, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            # Compute the critic loss as Q(s,a) - V(s)
            td_error = tf.math.subtract(states_actions_values, states_values) 
            critic_loss = tf.math.square(td_error)
  
            # Compute the critic gradient and update the network
            critic_grad = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    def train(self, tracker, n_episodes, verbose, params, offpolicy):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., gamma)
            offpolicy (bool): for the off-policy implementation

        Returns:
            None
        """
        
        mean_reward = deque(maxlen=100)

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                action, prob = self.get_action(state)
                obs_state, obs_reward, done, _ = self.env.step(action)

                self.buffer.store(state, 
                    action, 
                    prob, 
                    obs_reward, 
                    obs_state, 
                    1 - int(done)
                )

                ep_reward += obs_reward
                steps += 1

                state = obs_state

                if offpolicy:
                    self.update(params['gamma'], 
                        offpolicy, 
                        params['batch_size']
                    )

                if done: break  


            if not offpolicy:
                self.update(params['gamma'], 
                    offpolicy, 
                    params['batch_size']
                )

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        
