"""Tracker script for saving the model and the training stats

This instantiates the Tracker and manages it
"""

import os
import numpy as np

class Tracker:
    """
    A class used to represent the stats Tracker
    """

    def __init__(self, env_name, tag, seed, params, metrics):
        """Gets the training details and initiate the Tracker

        Args:
            env_name (str): gym environment name
            tag (str): algorithm name
            seed (int): training seed
            params (dict): dnn parameters for the .csv name
            metrics (list): metrics to save in the .csv
        """

        self.save_tag = env_name + \
            '_' + tag + \
            '_seed' + str(seed) + \
            '_' + str(params['actor']['h_layers']) + \
            'x' + str(params['actor']['h_size'])

        folder_name = 'results/'
        if not os.path.exists(folder_name): os.makedirs(folder_name)

        self.metric_save = folder_name + "metrics/"
        self.model_save = folder_name + "models/"

        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)

        np.savetxt(self.metric_save + self.save_tag + '.csv', 
            [metrics], 
            delimiter=',', 
            fmt='%s')

        self.metrics = []
        self.len_metrics = len(metrics)
        
    def update(self, metrics):
        """Store the training metrics

        Args:
            metrics (list): metrics to add to the Tracker
           
        Returns:
            None
        """

        assert self.len_metrics == len(metrics)
        self.metrics.append(metrics)

    def save_metrics(self):
        """Save the .csv

        Args:
            None

        Returns:
            None
        """
        
        with open(self.metric_save + self.save_tag + '.csv', 'a') as f:
            np.savetxt(f, self.metrics, delimiter=',', fmt='%s')
            self.metrics = []

    def save_model(self, model, epoch, success):
        """Save the model

        Args:
            model (Model): model to save
            epoch (int): epoch of the saved model
            success (int): performance of the saved model
                       
        Returns:
            None
        """

        model.save(self.model_save + self.save_tag + \
            '_epoch' + str(epoch) + \
            '_success' + str(success) + \
            '.h5')