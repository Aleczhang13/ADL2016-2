import numpy as np
import os
import random
from state import State
from ale_python_interface import ALEInterface
import cv2

# Terminology in this class:
#   Episode: the span of one game life
#   Game: an ALE game (e.g. in space invaders == 3 Episodes or 3 Lives)
#   Frame: An ALE frame (e.g. 60 fps)
#   Step: An Environment step (e.g. covers 4 frames)
#
class AtariEnvironment:
    
    def __init__(self, args, outputDir):
        
        #self.outputDir = outputDir
        self.screenCaptureFrequency = args.screen_capture_freq
        self.windowname = "preview"
        
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 123456)
        random.seed(123456)
        # Fix https://groups.google.com/forum/#!topic/deep-q-learning/p4FAIaabwlo
        self.ale.setFloat(b'repeat_action_probability', 0.0)

        # Load the ROM file
        self.ale.loadROM(args.rom)

        self.actionSet = self.ale.getMinimalActionSet()
        self.gameNumber = 0
        self.stepNumber = 0
        self.resetGame()
        cv2.startWindowThread()
        cv2.namedWindow(self.windowname)

    def getNumActions(self):
        return len(self.actionSet)

    def getState(self):
        return self.state
    
    def getGameNumber(self):
        return self.gameNumber
    
    def getFrameNumber(self):
        return self.ale.getFrameNumber()
    
    def getEpisodeFrameNumber(self):
        return self.ale.getEpisodeFrameNumber()
    
    def getEpisodeStepNumber(self):
        return self.episodeStepNumber
    
    def getStepNumber(self):
        return self.stepNumber
    
    def getGameScore(self):
        return self.gameScore

    def isGameOver(self):
        return self.ale.game_over()

    def step(self, action):
        previousLives = self.ale.lives()
        reward = 0
        isTerminal = 0
        self.stepNumber += 1
        self.episodeStepNumber += 1
        for i in range(4):
            prevScreenRGB = self.ale.getScreenRGB()
            reward += self.ale.act(self.actionSet[action])
            screenRGB = self.ale.getScreenRGB()
            if self.ale.lives() < previousLives or self.ale.game_over():
                isTerminal = 1
                break
        maxedScreen = np.maximum(screenRGB, prevScreenRGB)
        self.state = self.state.stateByAddingScreen(maxedScreen, self.ale.getFrameNumber())
        cv2.imshow(self.windowname,screenRGB)
        self.gameScore += reward
        return reward, self.state, isTerminal

    def resetGame(self):
        if self.ale.game_over():
            self.gameNumber += 1
        self.ale.reset_game()
        self.state = State().stateByAddingScreen(self.ale.getScreenRGB(), self.ale.getFrameNumber())
        self.gameScore = 0
        self.episodeStepNumber = 0 # environment steps vs ALE frames.  Will probably be 4*frame number
