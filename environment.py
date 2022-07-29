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
        initState = np.zeros( (mapSize, mapSize) )

        # Each map is stored as a numpy array, where
        # 1 indicates the block, and 2 indicates the
        # agent's position
        initState[blockRow][blockCol] = 1
        initState[-1][agentCol] = 2

        return initState

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
    # to values 0, 1, and 2 respectively. Returns -1, 0, 1, which correspond
    # to loss, no result or success respectively
    def act(self, action):
        currState = self.states[-1]

        # Compute current positions
        currAgentColumn = 0
        currBlockRow = 0
        currBlockColumn = 0
        for col, val in enumerate(currState[-1]):
            if val == 2:
                currAgentColumn = col

        for row, numpyRow in enumerate(currState):
            for col, val in enumerate(numpyRow):
                if val == 1:
                    currBlockRow = row
                    currBlockColumn = col

        # Update agent column based on input
        currAgentColumn += (action - 1)

        # Update block row based due to tstep
        currBlockRow += 1

        # Create new state, and add to current list
        newState = Game._numpyFromPosStates(currBlockColumn, currBlockRow, currAgentColumn, self.mapSize)

        self.states.append(newState)

        # If the current block row is the bottom 1, compute if we won or not and return that
        if currBlockRow == self.mapSize - 1:
            if currBlockColumn == currAgentColumn:
                return 1
            else:
                return -1
        # Otherwise, return 0
        else:
            return 0