"""DDPG agent script 

This manages the training phase of the off-policy DDPG.
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

class DDPG:
    """
    Class for the DDPG agent
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
        self.actor_tg = DeepNetwork.build(env, params['actor'], actor=True, name='actor_tg')
        self.actor_tg.set_weights(self.actor.get_weights())
        self.actor_optimizer = Adam()

        self.critic = DeepNetwork.build(env, params['critic'], name='critic')
        self.critic_tg = DeepNetwork.build(env, params['critic'], name='critic_tg')
        self.critic_tg.set_weights(self.critic.get_weights())
        self.critic_optimizer = Adam()

        self.buffer = Buffer(params['buffer']['size'])
        
    def get_action(self, state, std):
        """Get the action to perform

        Args:
            state (list): agent current state
            std (float): action noise

        Returns:
            action (float): sampled actions to perform
        """

        action = self.actor(np.array([state])).numpy()[0]
        action += np.random.normal(scale=std**2)

        return action

    def update(self, gamma, batch_size):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C

        Returns:
            None
        """

        batch_size = min(self.buffer.size, batch_size)
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.update_continuous(gamma, states, actions, rewards, obs_states, dones)

    def update_continuous(self, gamma, states, actions, rewards, obs_states, dones):
        """It can be seen as a continuos DQN. The actor tries to max the state-action values given by the critic: ∇θJ(θ) = ∇μ(s|θ)(Q(s,μ(s|θ)) ∇θ(μ(s|θ)) 
        The critic estimates Q, and is updated in DQN fashion, minimizing mse Q(s',μ(s'|θ)) - Q(s, a)

        Args:
            gamma (float): discount factor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """
        
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            # Compute the critic target Q(s', μ(s'|θ))
            target_actions = self.actor_tg(obs_states)
            obs_states_values = self.critic_tg([obs_states, target_actions]).numpy()
            critic_targets = rewards + gamma * obs_states_values * dones

            # Compute the critic value Q(s, a)
            critic_values = self.critic([states, actions])

            # Compute the critic loss as Q(s,a) - V(s)
            td_errors = tf.math.subtract(critic_targets, critic_values)
            td_errors = tf.math.square(td_errors)
            critic_loss = tf.math.reduce_mean(td_errors)

            # Compute the critic gradient and update the network
            critic_grad = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
            
            # Compute Q(s,μ(s|θ) and μ(s|θ) 
            actions = self.actor(states)
            actor_values = self.critic([states, actions])

            # Compute the actor objective, i.e., max the obtained values
            actor_objective = -tf.math.reduce_mean(actor_values)

            # Compute the actor gradient and update the network
            actor_grad = tape_a.gradient(actor_objective, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

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

        tau, std, std_scale = hyperp['tau'], hyperp['std'], hyperp['std_scale']
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
            
                # Pendulum-v0 requires a huge amount of updates. Hence, we do it at every step
                # This update pattern returns -100 reward in 25 episodes in Pendulum.
                # In a scenario with significant less steps per episodes (e.g. LunarLanderContinuous), the updates could be performed after every episode
                if steps > 100:
                    self.update(
                        params['gamma'], 
                        params['buffer']['batch']
                    )
                    self.polyak_update(self.actor.variables, self.actor_tg.variables, tau)
                    self.polyak_update(self.critic.variables, self.critic_tg.variables, tau)

                if done: break  

            if std_scale: std = max(std_min, std * std_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        


   