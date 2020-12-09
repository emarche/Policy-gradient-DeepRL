"""Launch file for the continuous SAC algorithm

This script instantiate the gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml

from agent import SAC
from utils.tracker import Tracker

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

if not cfg['setup']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Gym env', default=cfg['train']['name'])
parser.add_argument('-epochs', type=int, help='Epochs', default=cfg['train']['n_episodes'])
parser.add_argument('-verbose', type=int, help='Save stats freq', default=cfg['train']['verbose'])
parser.add_argument('-tau', type=float, help='Target net τ', default=cfg['agent']['tau'])
parser.add_argument('-std', type=float, help='σ for noise', default=cfg['agent']['std'])
parser.add_argument('-std_scale', type=float, help='σ scaling', default=cfg['agent']['std_scale'])

def main(params):
    config = vars(parser.parse_args())

    env = gym.make(config['env'])
    env.seed(seed)
    
    agent = SAC(env, cfg['agent'])
    tag = 'SAC_Continuous'

    # Initiate the tracker for stats
    tracker = Tracker(
        env.unwrapped.spec.id,
        tag,
        seed,
        cfg['agent'], 
        ['Epoch', 'Ep_Reward']
    )

    # Train the agent
    agent.train(
        tracker,
        n_episodes=config['epochs'], 
        verbose=config['verbose'],
        params=cfg['agent'],
        hyperp=config
    )

if __name__ == "__main__":
    main(cfg)
