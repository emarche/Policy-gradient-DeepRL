"""SAC agent script 

This manages the training phase of the off-policy SAC with fixed trade-off coefficient α.
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

class SAC:
    """
    Class for the SAC agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)

        Returns:
            None
        """
        
        self.env = env

        self.actor = DeepNetwork.build(env, params['actor'], actor=True, name='actor')
        self.actor_opt = Adam()

        self.critic1 = DeepNetwork.build(env, params['critic'], name='critic1')
        self.critic2 = DeepNetwork.build(env, params['critic'], name='critic2')
        self.critic1_tg = DeepNetwork.build(env, params['critic'], name='critic1_tg')
        self.critic2_tg = DeepNetwork.build(env, params['critic'], name='critic2_tg')
        self.critic1_tg.set_weights(self.critic1.get_weights())
        self.critic2_tg.set_weights(self.critic2.get_weights())
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        
        self.buffer = Buffer(params['buffer']['size'])

    def get_action(self, state, std):
        """Get the action to perform

        Args:
            state (list): agent current state
            std (float): Gaussian distribution std for action selection

        Returns:
            action (float): sampled actions to perform
            mu (float): sampled mu from the network

        """

        mu = self.actor(np.array([state])).numpy()[0]
        action = np.random.normal(loc=mu, scale=std)
        return action
    
    def update(self, gamma, batch_size, std):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C
            std (float): Gaussian distribution std for action selection

        Returns:
            None
        """
        
        batch_size = min(self.buffer.size, batch_size)
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.update_continuous(gamma, std, \
            states, actions, rewards, obs_states, dones)

    def update_continuous(self, gamma, std, states, actions, rewards, obs_states, dones):
        """Improved version of TD3. It learns two Q-functions, and uses the smaller Q to form the targets. It uses an entropy term in the update of both the critic and the actor. 
        It uses the current policy to sample the actions. 
        The actor tries to max the state-action values given by the critic, sampling the action 
        a_squash(s|θ) = tanh(μ(s|θ) + δ(s) * N(0, I)), where μ is the network output (i.e., the mean of the Gaussian). So it remove the stochasticity from the action selection, and it squash the actions with a tanh. The policy is then updated with: (minQ(s,a_squash(s|θ)) - α log π(a_squash(s|θ)|θ)) using the min Q between the two critics.
        The critic estimates the min Q_target using the current policy to estimate the next best action. It is then updated in DQN fashion, minimizing mse (Q(s, a) - y). Where y = r + γ * done * (min Q_targ(s', π(s'|θ)) - α log (π(s'|θ)|s')))

        Args:
            gamma (float): discount factor
            std (float): Gaussian distribution std for action selection
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """

        with tf.GradientTape() as tape_c1, tf.GradientTape() as tape_c2, tf.GradientTape() as tape_a:
            # Compute π(s'|θ) for the critic target
            mu = self.actor(obs_states)
            tg_actions = np.random.normal(loc=mu, scale=std)

            # Compute the Q_targ(s',π(s'|θ))
            tg1_values = self.critic1_tg([obs_states, tg_actions]).numpy()
            tg2_values = self.critic2_tg([obs_states, tg_actions]).numpy()
            min_tg_values = tf.math.minimum(tg1_values, tg2_values)

            # Compute α log π(π(s'|θ)|s')
            gauss_d = std * tf.sqrt(2 * np.pi)
            gauss_n = tf.math.exp(-0.5 * ((tg_actions - mu) / std)**2)
            gauss_n = tf.cast(gauss_n, dtype=np.float32)
            gauss_p = tf.math.divide(gauss_n, gauss_d)
            # We combine the contribution of the actions in case of |actions| > 1
            # It works also without this, but it optimize the learning process
            gauss_p = tf.math.reduce_mean(gauss_p, axis=1, keepdims=True) 
            log_p = tf.math.log(gauss_p)
            log_p *= 0.2    # α = 0.2

            # Compute the critic target
            critic_targets = rewards + gamma * min_tg_values * dones
            critic_targets = tf.math.subtract(critic_targets, log_p).numpy()

            # Compute Q(s, a)
            states_values1 = self.critic1([states, actions])
            states_values2 = self.critic2([states, actions])

            # Compute the critics loss as target - V(s) using the min target
            td_error1 = tf.math.subtract(critic_targets, states_values1) 
            critic1_loss = tf.math.square(td_error1) 
            critic1_loss = tf.math.reduce_mean(critic1_loss)

            td_error2 = tf.math.subtract(critic_targets, states_values2) 
            critic2_loss = tf.math.square(td_error2) 
            critic2_loss = tf.math.reduce_mean(critic2_loss)

            # Compute the critics gradient and update the network
            critic1_grad = tape_c1.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_opt.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))
            critic2_grad = tape_c2.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_opt.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))

            # Compute a_squash(s|θ) = tanh(μ(s|θ) + δ(s) * N(0, I))   
            mu = self.actor(obs_states)
            action_squashed = mu + tf.random.normal(shape=mu.shape)
            action_squashed = tf.math.tanh(action_squashed)

            # Compute α log π(a_squash(s|θ)|θ)
            gauss_n = tf.math.exp(-0.5 * ((action_squashed - mu) / std)**2)
            gauss_n = tf.cast(gauss_n, dtype=np.float32)
            gauss_p = tf.math.divide(gauss_n, gauss_d)
            # We combine the contribution of the actions in case of |actions| > 1
            # It works also without this, but it optimize the learning process
            gauss_p = tf.math.reduce_mean(gauss_p, axis=1, keepdims=True) 
            log_p = tf.math.log(gauss_p)
            log_p *= 0.2    # α = 0.2    

            # Compute min Q(s,a_squash(s|θ))
            states_values1 = self.critic1([states, action_squashed]).numpy()
            states_values2 = self.critic2([states, action_squashed]).numpy()
            min_tg_values = tf.math.minimum(tg1_values, tg2_values)

            # Compute actor objective max (minQ(s,a_squash(s|θ)) - α log π(a_squash(s|θ)|θ))
            actor_objective = tf.math.subtract(min_tg_values, log_p)
            actor_objective = -tf.math.reduce_mean(actor_objective)

            # Compute the actor gradient and update the network
            actor_grad = tape_a.gradient(actor_objective, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def polyak_update(self, weights, target_weights, tau):
        """Polyak update for the target networks

        Args:
            weights (list): network weights
            target_weights (list): target network weights
            tau (float): controls the update rate

        Returns:
            None
        """

        for (w, tw) in zip(weights, target_weights):
            tw.assign(w * tau + tw * (1 - tau))

    def train(self, tracker, n_episodes, verbose, params, hyperp):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., the critic's gamma)
            hyperp (dict): algorithmic specific values (e.g., tau)

        Returns:
            None
        """

        mean_reward = deque(maxlen=100)

        tau = hyperp['tau']
        std, std_scale = hyperp['std'], hyperp['std_scale']
        std_decay, std_min = params['std_decay'], params['std_min']

        steps = 0
        for e in range(n_episodes):
            ep_reward = 0
        
            state = self.env.reset()

            while True:
                action = self.get_action(state, std)
                obs_state, obs_reward, done, _ = self.env.step(action)

                self.buffer.store(state, 
                    action,
                    obs_reward, 
                    obs_state, 
                    1 - int(done)
                )

                ep_reward += obs_reward
                steps += 1

                state = obs_state
            
                if steps >= 100:
                    self.update(                        
                        params['gamma'], 
                        params['buffer']['batch'],
                        std
                    )
                    self.polyak_update(self.critic1.variables, self.critic1_tg.variables, tau)
                    self.polyak_update(self.critic2.variables, self.critic2_tg.variables, tau)      

                if done: break  

            if std_scale: 
                std = max(std_min, std * std_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        
