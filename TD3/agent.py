"""TD3 agent script 

This manages the training phase of the off-policy TD3.
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

class TD3:
    """
    Class for the TD3 agent
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
            std (float): action noise

        Returns:
            action (float): sampled actions to perform
        """

        action = self.actor(np.array([state])).numpy()[0]
        action += np.random.normal(scale=std**2)

        return action
    
    def update(self, gamma, batch_size, std_tg, actor_update=False):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C
            std_tg (float): std for target network
            actor_update (bool): wether to update the actor

        Returns:
            None
        """
        
        batch_size = min(self.buffer.size, batch_size)
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.update_continuous(gamma, std_tg, actor_update, \
            states, actions, rewards, obs_states, dones)

    def update_continuous(self, gamma, std_tg, actor_update, states, actions, rewards, obs_states, dones):
        """Improved version of DDPG. It learns two Q-functions, and uses the smaller Q to form the targets. It updates the policy (and target networks) less frequently than the critics. It adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action. 
        The actor tries to max the state-action values given by the critic: ∇θJ(θ) = ∇μ(s|θ)(Q(s,μ(s|θ)) ∇θ(μ(s|θ)) using the min Q between the two
        The critic estimates Q, and is updated in DQN fashion, minimizing mse Q(s',μ(s'|θ)) - Q(s, a), where μ(s'|θ) is summed with clipped target noise 

        Args:
            gamma (float): discount factor
            std_tg (float): std for target network
            actor_update (bool): wether to update the actor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """

        with tf.GradientTape() as tape_c1, tf.GradientTape() as tape_c2:
            # Compute the clipped target noise (with fixed values 0.5)
            tg_noise = np.array([[np.clip(np.random.normal(scale=std_tg), -0.5, 0.5)] for _ in range(len(states))])

            # Compute μ(s'|θ) and add the clipped noise
            tg_actions = self.actor_tg(obs_states)
            tg_actions = np.clip(tg_actions + tg_noise, \
                self.env.action_space.low[0], self.env.action_space.high[0])

            # Compute the two Q(s',μ(s'|θ))
            tg1_values = self.critic1_tg([obs_states, tg_actions]).numpy()
            tg2_values = self.critic2_tg([obs_states, tg_actions]).numpy()
            min_tg_values = tf.math.minimum(tg1_values, tg2_values)

            # Compute the critic target
            critic_targets = rewards + gamma * min_tg_values * dones
            
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

        if actor_update:
            with tf.GradientTape() as tape_a:
                # Compute Q(s,μ(s|θ) and μ(s|θ) 
                # TD3 always uses critic1 for the actor's Q
                actions = self.actor(states)
                states_values = self.critic1([states, actions])

                # Compute the actor objective, i.e., max the obtained values
                actor_objective = -tf.math.reduce_mean(states_values)

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
        update_delay = params['update_delay'] 
        std, std_tg, std_scale = hyperp['std'], hyperp['std_tg'], hyperp['std_scale']
        std_decay, std_min = params['std_decay'], params['std_min']

        steps = 0
        
        for e in range(n_episodes):
            ep_reward =  0  
        
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
            
                if steps % update_delay == 0 and steps > 100:
                    self.update(
                        params['gamma'], 
                        params['buffer']['batch'],
                        std_tg,
                        actor_update=True
                    ) 
                    self.polyak_update(self.actor.variables, self.actor_tg.variables, tau)
                    self.polyak_update(self.critic1.variables, self.critic1_tg.variables, tau)
                    self.polyak_update(self.critic2.variables, self.critic2_tg.variables, tau)
                elif steps % update_delay != 0 and steps > 100: 
                    self.update(                        
                        params['gamma'], 
                        params['buffer']['batch'],
                        std_tg
                    )
                
                if done: break  

            if std_scale: 
                std = max(std_min, std * std_decay)
                std_tg = max(std_min, std_tg * std_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        
