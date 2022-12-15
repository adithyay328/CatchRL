# Implementation of the catch game environment / map
import random

import numpy as np

# This class stores the state for a single game instance,
# and has APIs to allow the agent to manipulate it
class Game:
    @staticmethod
    # Method that creates a game map with position states
    def _numpyFromPosStates(blockCol, blockRow, agentCol, mapSize):
        # Shape represents (row, column) : first index is row
        initState = np.zeros( (mapSize + 2, mapSize) )

        # Each map is stored as a numpy array, where
        # 1 indicates the block, and 2 indicates the
        # agent's position
        initState[blockRow][blockCol] = 1
        initState[-1][agentCol] = 2

        return initState

    """This method is like the numpy from pos states function,
    but instead outputs a depth 2 map, with the first layer
    one-hot encoded with the block row, while the second layer
    is one-hot encoded with the agent row."""
    @staticmethod
    def _depthTwoFromPosStates(blockCol, blockRow, agentCol, mapSize):
      result = np.zeros( (2, mapSize + 2, mapSize) )
      
      # Stroing one-hot values
      result[0][blockRow][blockCol] = 1
      result[1][-1][agentCol] = 1

      return result
    
    """Converts a depth two array into a depth one array like the one from
    numpy from pos states."""
    @staticmethod
    def _oneDeepArrayFromDepthTwoArray(depthTwoArray):
      # Multiply the second layer by 2 to convert the agent's position into a 2
      depthTwoArray[1] = depthTwoArray[1] * 2

      # Now just add the 2 layers together
      oneDeep = depthTwoArray[0] + depthTwoArray[1]

      return oneDeep
    
    # Takes in a numpy state array of the environment.
    # and returns a dictionary of positions with keys:
    # blockCol, blockRow, agentCol
    @staticmethod
    def posDictFromNumpyArray(stateArray):
        currAgentColumn = 0
        currBlockRow = 0
        currBlockColumn = 0
        for col, val in enumerate(stateArray[-1]):
            if val == 2:
                currAgentColumn = col

        for row, numpyRow in enumerate(stateArray):
            for col, val in enumerate(numpyRow):
                if val == 1:
                    currBlockRow = row
                    currBlockColumn = col

        return {
            "blockCol" : currBlockColumn,
            "blockRow" : currBlockRow,
            "agentCol" : currAgentColumn
        }


    # Map size indicates the width of the game map
    def __init__(self, mapSize):
        self.mapSize = mapSize

        # Stores a list of game states, where index maps
        # to timestep i.e. index 0 maps to t0
        self.states = []

        # By default, randomly initialize the map with
        # agent position and block; store at t0. The map
        # is square, as the block and agent can go to the same unit
        # at the same time step and win; wins are when the agent
        # and block are at the same column in the bottom row at some
        # timestep, and losses are when block is in bottom row
        # and the agent is in a different column

        # Generating random positions
        blockColumn = random.randint(0, mapSize - 1)
        agentColumn = random.randint(0, mapSize - 1)

        self.states.append(Game._numpyFromPosStates(blockColumn, 0, agentColumn, self.mapSize))
    
    # The way timesteps work here is that a new timestep
    # is only created when the agent makes a move. There
    # are 3 possible actions: left, none, right, which map
    # to values -1, 0, and 1 respectively. Returns -1, 0, 1, which correspond
    # to loss, no result or success respectively
    def act(self, action):
        currState = self.states[-1]

        # Compute current positions
        posDict = Game.posDictFromNumpyArray(currState)
        currAgentColumn = posDict["agentCol"]
        currBlockRow = posDict["blockRow"]
        currBlockColumn = posDict["blockCol"]

        # Update agent column based on input, if it's valid
        currAgentColumn += action
        if currAgentColumn == self.mapSize:
          currAgentColumn = self.mapSize - 1
        elif currAgentColumn == -1:
          currAgentColumn = 0

        # Update block row based due to tstep
        currBlockRow += 1

        # Create new state, and add to current list
        newState = Game._numpyFromPosStates(currBlockColumn, currBlockRow, currAgentColumn, self.mapSize)

        self.states.append(newState)

        # If the current block row is the bottom, compute if we won or not and return that
        if currBlockRow == self.states[-1].shape[0] - 1:
            if currBlockColumn == currAgentColumn:
                return 1
            else:
                return -1
        # Otherwise, return 0
        else:
            return 0