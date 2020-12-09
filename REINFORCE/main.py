"""Launch file for the discrete and continuous REINFORCE agent, with a baseline option

This script instantiate the gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml
from gym.spaces import Box

from agent import Reinforce
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
parser.add_argument('-baseline', type=bool, help='Use baseline', default=cfg['train']['baseline'])

def main():
    config = vars(parser.parse_args())

    env = gym.make(config['env'])
    env.seed(seed)

    # Discrete / Continuous agent based on the Gym env
    if (isinstance(env.action_space, Box)):
        agent = Reinforce(env, cfg['agent'], continuous=True)
        tag = 'Reinforce_Continuous'
    else:
        agent = Reinforce(env, cfg['agent'], continuous=False)
        tag = 'Reinforce_Discrete'

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
        baseline=config['baseline'],
    )

if __name__ == "__main__":
    main()
