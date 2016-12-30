import numpy as np
import scipy.ndimage as ndimage

class State:

    useCompression = False

    @staticmethod
    def setup(args):
        State.useCompression = args.compress_replay

    def stateByAddingScreen(self, screen):
        screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84, 1))

        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = self.screens[:3]
            newState.screens.insert(0, screen)
        else:
            newState.screens = [screen, screen, screen, screen]
        return newState
    def getScreens(self):
        s = self.screens
        return np.concatenate(s, axis=2)
