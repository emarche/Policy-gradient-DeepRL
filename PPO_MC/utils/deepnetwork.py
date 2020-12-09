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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
tf.random.set_seed(seed)

class DeepNetwork:
    """
    Class for the DNN creation
    """

    @staticmethod  
    def build(env, params, continuous=False, actor=False, name='model'):
        """Gets the DNN architecture and build it

        Args:
            env (gym): the gym env to take agent I/O
            params (dict): n° and size of hidden layers, print model for debug
            continuous (Bool): continuous or discrete environment
            actor (bool): wether to build the actor or the critic
            name (str): file name for the model

        Returns:
            model: the uncompiled DNN Model
        """

        input_size = env.observation_space.shape[0]

        if continuous: action_size = env.action_space.shape[0]
        else: action_size = env.action_space.n

        h_layers = params['h_layers']
        h_size = params['h_size']

        state_input = Input(shape=(input_size,), name='input_layer')
        h = state_input
        for i in range(h_layers):
            h = Dense(h_size, activation='relu', name='hidden_' + str(i))(h)

        if actor:
            if continuous:
                y = Dense(action_size, activation='tanh', name='output_layer')(h)  
            else:
                y = Dense(action_size, activation='softmax', name='output_layer')(h)
        else:
            y = Dense(1, activation='linear', name='critic_output_layer')(h)
   
        model = Model(inputs=state_input, outputs=y)

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

