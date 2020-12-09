"""PPO (TD) agent script

This manages the training phase of both the discrete and the continuous agents that can perform updates in TD fashion.
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

class PPO:
    """
    Class for the PPO (TD) agent
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

        self.actor = DeepNetwork.build(env, params['actor'], continuous, actor=True, name='actor')
        self.actor_optimizer = Adam()

        self.critic = DeepNetwork.build(env, params['critic'], continuous, name='critic')
        self.critic_optimizer = Adam()
      
        self.buffer = Buffer()

    def get_action(self, state, std=1.0):
        """Get the action to perform

        Args:
            state (list): agent current state
            std (float): std for the Gaussian in cc

        Returns:
            action (list): sampled action to perform, integers or floats
            mu (float): Gaussian's mean in cc
            probs[action] (float): prob of the selected action in dc
        """

        if self.continuous:
            mu = self.actor(np.array([state])).numpy()[0]
            action = np.random.normal(loc=mu, scale=std)
            return action, mu
        
        probs = self.actor(np.array([state])).numpy()[0]
        action = np.random.choice(self.env.action_space.n, p=probs)

        return action, probs[action]

    def update(self, batch_size, epochs, eps, gamma, std=1.0):
        """Prepare the samples and the cumulative reward to update the network

        Args:
            batch_size (int): batch size 
            epochs (int): n° of epochs to perform
            eps (float): clipping value for PPO-clip
            gamma (float): discount factor
            std (float): std for the Gaussian in cc

        Returns:
            None
        """

        states, probs, actions, rewards, obs_states, dones = self.buffer.sample()
        buffer = [[states[i], probs[i], actions[i], rewards[i], obs_states[i], dones[i]] for i in range(self.buffer.size)]

        np.random.shuffle(buffer)
        batch_size = min(self.buffer.size, batch_size)
        batches = np.array_split(np.array(buffer), int(len(buffer) / batch_size))

        if self.continuous:
            self.update_continuous(batches, epochs, eps, gamma, std)
        else:
            self.update_discrete(batches, epochs, eps, gamma)

        # After the update, we clear the buffer
        self.buffer.clear()

    def update_discrete(self, batches, epochs, eps, gamma):
        """PPO does not want the new policy to move too far away from the old one
        ∇θJ(θ) ≈ 1/N * ∑ [min(∑ ∇ ratio(θ)*A, clip(ratio(θ), 1-ε, 1+ε)*A] where the advantage A is computed as desired, and the ratio(θ) = π'θ(a∣s)/πθ(a∣s) is similar to the importance sampling. 
        In the discrete case, the πθ is directly given by the softmax. 

        Args:
            batches (list): list of minibatches
            epochs (int): n° of epochs to perform
            eps (float): clipping value for PPO-clip
            gamma (float): discount factor

        Returns:
            None
        """

        for _ in range(epochs):
            for minibatch in batches:
                states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
                old_probs = np.array([sample[1] for sample in minibatch], dtype=np.float32)
                actions = np.array([sample[2] for sample in minibatch])
                rewards = np.array([sample[3] for sample in minibatch], dtype=np.float32)
                obs_states = np.array([sample[4] for sample in minibatch], dtype=np.float32)
                dones = np.array([sample[5] for sample in minibatch])

                # The updates require shape (n° samples, len(metric))
                old_probs = old_probs.reshape(-1, 1)
                rewards = rewards.reshape(-1, 1)
                dones = dones.reshape(-1, 1)

                with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                    # Compute the advantage A as in TD
                    states_values = self.critic(states) # each element is an array with 1 value
                    obs_states_values = self.critic(obs_states)

                    # Both critic values and target are differentiable wrt critic. 
                    # We have to cast the target to a numpy array
                    # (i.e., we do not want our "label" to be diff)
                    advantages = rewards + gamma * obs_states_values.numpy() * dones
                    advantages = tf.math.subtract(advantages, states_values)

                    # Compute the ratio(θ): π'θ(a∣s)/πθ(a∣s)
                    probs = self.actor(states)
                    idxs = np.array([[int(i), int(action)] for i, action in enumerate(actions)])
                    action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

                    ratio = tf.math.divide(action_probs, old_probs + 1e-10) 

                    # Compute the two actor objectives of which we will take the min
                    actor_objective_1 = ratio * advantages
                    actor_objective_2 = tf.clip_by_value(ratio, 1 - eps, 1 + eps) * advantages

                    # Compute the entropy for exploration in dc
                    entropy = -tf.math.multiply(old_probs, tf.math.log(old_probs + 1e-10))
                    entropy = tf.reduce_mean(entropy)

                    # Compute the actor objective
                    actor_objective = tf.math.minimum(actor_objective_1, actor_objective_2) 
                    actor_loss = -tf.math.reduce_mean(actor_objective) 
                    actor_loss += 0.001 * entropy

                    # Compute the actor gradient and update the network
                    actor_grad = tape_a.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                    # Compute the critic loss in TD
                    critic_loss = tf.math.square(advantages)
                    critic_loss = tf.math.reduce_mean(critic_loss)

                    # Compute the critic gradient and update the network
                    critic_grad = tape_c.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    def update_continuous(self, batches, epochs, eps, gamma, std):
        """PPO does not want the new policy to move too far away from the old one
        ∇θJ(θ) ≈ 1/N * ∑ [min(∑ ∇ ratio(θ)*A, clip(ratio(θ), 1-ε, 1+ε)*A] where the advantage A is computed as desired, and the ratio(θ) = π'θ(a∣s)/πθ(a∣s) is similar to the importance sampling. 
        In the continuous case the network outputs mu for a Gaussian distribution that we use as πθ

        Args:
            batches (list): list of minibatches
            epochs (int): n° of epochs to perform
            eps (float): clipping value for PPO-clip
            gamma (float): discount factor
            std (float): std for the Gaussian in cc

        Returns:
            None
        """

        for _ in range(epochs):
            for minibatch in batches:
                states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
                old_mu = np.array([sample[1] for sample in minibatch], dtype=np.float32)
                actions = np.array([sample[2] for sample in minibatch])
                rewards = np.array([sample[3] for sample in minibatch], dtype=np.float32)
                obs_states = np.array([sample[4] for sample in minibatch], dtype=np.float32)
                dones = np.array([sample[5] for sample in minibatch])

                # The updates require shape (n° samples, len(metric))
                rewards = rewards.reshape(-1, 1)
                dones = dones.reshape(-1, 1)

                with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                    # Compute the advantage A as in TD
                    states_values = self.critic(states) # each element is an array with 1 value
                    obs_states_values = self.critic(obs_states)

                    # Both critic values and target are differentiable wrt critic. 
                    # We have to cast the target to a numpy array
                    # (i.e., we do not want our "label" to be diff)
                    advantages = rewards + gamma * obs_states_values.numpy() * dones
                    advantages = tf.math.subtract(advantages, states_values)

                    # Compute the ratio(θ): π'θ(a∣s)/πθ(a∣s)
                    gauss_d = std * tf.sqrt(2 * np.pi)

                    # Compute πθ(a∣s)
                    gauss_old_n = tf.math.exp(-0.5 * ((actions - old_mu) / std)**2)
                    gauss_old_n = tf.cast(gauss_old_n, dtype=np.float32)
                    gauss_old_p = tf.math.divide(gauss_old_n, gauss_d)
                    # We combine the contribution of the actions in case of |actions| > 1
                    # It works also without this, but it optimize the learning process
                    gauss_old_p = tf.math.reduce_mean(gauss_old_p, axis=1, keepdims=True)  

                    # Compute π'θ(a∣s)
                    mu = self.actor(states)
                    gauss_n = tf.math.exp(-0.5 * ((actions - mu) / std)**2)
                    gauss_n = tf.cast(gauss_n, dtype=np.float32)
                    gauss_p = tf.math.divide(gauss_n, gauss_d)
                    gauss_p = tf.math.reduce_mean(gauss_p, axis=1, keepdims=True)  

                    ratio = tf.math.divide(gauss_p, gauss_old_p + 1e-10) 

                    # Compute the two actor objectives of which we will take the min
                    actor_objective_1 = ratio * advantages
                    actor_objective_2 = tf.clip_by_value(ratio, 1 - eps, 1 + eps) * advantages

                    # Compute the actor objective
                    actor_objective = tf.math.minimum(actor_objective_1, actor_objective_2) 
                    actor_loss = -tf.math.reduce_mean(actor_objective) 

                    # Compute the actor gradient and update the network
                    actor_grad = tape_a.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                    # Compute the critic loss in TD
                    critic_loss = tf.math.square(advantages)
                    critic_loss = tf.math.reduce_mean(critic_loss)

                    # Compute the critic gradient and update the network
                    critic_grad = tape_c.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    def train(self, tracker, n_episodes, verbose, params, hyperp):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., std for the Gaussian)
            hyperp (dict): algorithmic specific values (e.g., scaling of the std)

        Returns:
            None
        """

        mean_reward = deque(maxlen=100)

        std, std_scale = hyperp['std'], hyperp['std_scale']
        std_decay, std_min = params['std_decay'], params['std_min']

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                if self.continuous: 
                    action, mu = self.get_action(state, std)
                else: 
                    action, prob = self.get_action(state)
                
                obs_state, obs_reward, done, _ = self.env.step(action)

                if self.continuous:
                    self.buffer.store(state, mu, action, obs_reward, obs_state, 1 - int(done))
                else:
                    self.buffer.store(state, prob, action, obs_reward, obs_state, 1 - int(done))

                ep_reward += obs_reward
                steps += 1
                if done: break  

                state = obs_state
    
            if e % params['update_freq'] == 0:
                self.update(
                    params['buffer']['batch'],
                    params['n_epochs'],
                    params['eps_clip'],
                    params['gamma'],
                    std
                )

                if std_scale: std = max(std_min, std * std_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        