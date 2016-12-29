import numpy as np
import os
import random
import replay
import time
import argparse
import dqn as DQN
from atari_environment import AtariEnvironment
from state import State

class Agent(object):
    def __init__(self, sess, min_action_set):
        self.sess = sess
        self.min_action_set = min_action_set
        self.build_dqn()

    def build_dqn(self):
        self.state = None
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-epoch-steps", type=int, default=250000, help="how many steps (=4 frames) to run during a training epoch (approx -- will finish current game)")
        parser.add_argument("--eval-epoch-steps", type=int, default=125000, help="how many steps (=4 frames) to run during an eval epoch (approx -- will finish current game)")
        parser.add_argument("--replay-capacity", type=int, default=1000000, help="how many states to store for future training")
        parser.add_argument("--prioritized-replay", action='store_true', help="Prioritize interesting states when training (e.g. terminal or non zero rewards)")
        parser.add_argument("--compress-replay", action='store_true', help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
        parser.add_argument("--normalize-weights", action='store_true', default=True, help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
        parser.add_argument("--screen-capture-freq", type=int, default=250, help="record screens for a game this often")
        parser.add_argument("--save-model-freq", type=int, default=10000, help="save the model once per 10000 training sessions")
        parser.add_argument("--observation-steps", type=int, default=50000, help="train only after this many stesp (=4 frames)")
        parser.add_argument("--learning-rate", type=float, default=0.00025, help="learning rate (step size for optimization algo)")
        parser.add_argument("--target-model-update-freq", type=int, default=10000, help="how often (in steps) to update the target model.  Note nature paper says this is in 'number of parameter updates' but their code says steps. see tinyurl.com/hokp4y8")
        parser.add_argument("a", help="rom file to run")
        parser.add_argument("b", help="rom file to run")
        args = parser.parse_args()
        print 'Arguments: %s' % (args)
        baseOutputDir = "./models/"
        State.setup(args)
        self.dqn = DQN.DeepQNetwork(4, baseOutputDir, args)
        replayMemory = replay.ReplayMemory(args)

    def getSetting(self):
        """
        # TODO
            You can only modify these three parameters.
            Adding any other parameters are not allowed.
            1. action_repeat: number of time for repeating the same action 
            2. screen_type: return 0 for RGB; return 1 for GrayScale
        """
        action_repeat = 4
        screen_type = 0
        return action_repeat, screen_type

    def play(self, screen):
       stateReward = 0
       # Choose next action
       epsilon = 0.05
       if self.state is None or random.random() > (1 - epsilon):
           action = random.randrange(4)
           self.state = State().stateByAddingScreen(screen)
           return self.min_action_set[action]
       self.state = self.state.stateByAddingScreen(screen)
       screens = np.reshape(self.state.getScreens(), (1, 84, 84, 4))
       action = self.dqn.inference(screens)
       return self.min_action_set[action]

    def get_state(screen):
        return state
