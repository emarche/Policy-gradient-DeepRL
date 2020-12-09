"""DNN builder script

This manages the DNN creation and printing for the agent
"""

import yaml

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
tf.random.set_seed(seed)

class DeepNetwork:
    """
    Class for the DNN creation of both the actor and the critic
    """

    @staticmethod  
    def build(env, params, actor=False, name='model'):
        """Gets the DNN architecture and build it

        Args:
            env (gym): the gym env to take agent I/O
            params (dict): n° and size of hidden layers, print model for debug
            actor (bool): wether to build the actor or the critic
            name (str): file name for the model

        Returns:
            model: the uncompiled DNN Model
        """

        input_size = env.observation_space.shape[0]

        action_size = env.action_space.shape[0]
        # Some envs does not env symmetric bounds for the actions!
        action_range = env.action_space.high[0]

        state_input = Input(shape=(input_size,), name='input_state_layer')
        h_state = state_input
        
        if not actor:
            for i in range(params['h_state_layers']):
                h_state = Dense(params['h_state_size'], activation='relu', name='hidden_state_' + str(i))(h_state)

            action_input = Input(shape=(action_size,), name='input_action_layer')
            h_action = action_input
            for i in range(params['h_action_layers']):
                h_action = Dense(params['h_action_size'], activation='relu', name='hidden_action_' + str(i))(h_action)

            h = Concatenate()([h_state, h_action])

        h_size = params['h_size']
        h_layers = params['h_layers']

        if actor: h = h_state
        for i in range(h_layers):
            h = Dense(h_size, activation='relu', name='hidden_' + str(i))(h)

        if actor:
            # Pendulum requires to explore its actions starting from low values
            k_init = tf.random_normal_initializer()
            if env.unwrapped.spec.id == 'Pendulum-v0':
                k_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            
            y = Dense(action_size, activation='tanh', name='actor_output_layer', kernel_initializer=k_init)(h)
            y = Lambda(lambda i: i * action_range, name='lamba_output_layer')(y)
            model = Model(inputs=state_input, outputs=y)

        else:
            y = Dense(action_size, activation='linear', name='critic_output_layer')(h)
            model = Model(inputs=[state_input, action_input], outputs=y)

        # PNG with the architecture and summary
        if params['print_model']:
            plot_model(model, to_file=name + '.png', show_shapes=True)    
            model.summary()

        return model

    @staticmethod  
    def print_weights(model):
        """Gets the model and print its weights layer by layer

        Args:
            model (Model): model to print
           
        Returns:
            None
        """

        model.summary()
        print("Configuration: " + str(model.get_config()))
        for layer in model.layers:
            print(layer.name)
            print(layer.get_weights())  
