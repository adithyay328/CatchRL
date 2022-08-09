# Implements a q-table based agent.
import random
import time
import sys

import numpy as np

from environment import Game
import q_base

class QTableAgent:
  def __init__(self):
    # Map size is set here so that the q-table
    # and map match in terms of # possible states
    self.mapSize = 10
  
    # For any given map size, there are mapSize * (mapSize + 2) states representing the
    # possible positions of the block, and mapSize * 1 states representing the possible
    # positions of the agent. Multiplied together, we end up with mapSize * mapSize * (mapSize + 2)
    # possible states. As well, for each state, there
    # are 3 possible actions: -1(go left), 0(do nothing), 1(go right).
    # To make q-table manipulation inuitive, the shape of the qtable is defined
    # in a similar way to the game state table:
    
    # (block row, block col, agent col, action-reward)
    self.qTable = np.zeros( (self.mapSize + 2, self.mapSize, self.mapSize, 3) )

    # When initializing, the one thing we need to do is to add a huge
    # penalty to moves that move the agent off the board.
    for blockRow in range(0, self.mapSize + 2):
      for blockCol in range(0, self.mapSize):
        # Just a big negative number to penalize it
        PENALTY_VALUE = -1
        # Preventing agent from moving left
        # in position 0
        self.qTable[blockRow][blockCol][0][0] = PENALTY_VALUE    
        # Preventing agent from moving right in position
        # mapSize - 1
        self.qTable[blockRow][blockCol][-1][-1] = PENALTY_VALUE

  """
  Let the Q-Table agent learn on the game, with some
  paramaters.

  Parameters
  ----------
  epsilon: int
      Represents the probability that we randomly
      select the action
  epsilonDecay: float
      The decay of epsilon per epoch
  epochs: int
      The number of epochs to train for
  discount: float
      The discount factor used in Bellman equation
  LR: float
      The learning rate used in the Bellman equation
  """
  def learn(self, epsilon, epsilonDecay, epochs, discount, LR):
    for epoch in range(epochs):
      # A new game instance for this epoch
      game = Game(self.mapSize)

      # Whether or not the game is finished
      finished = False
      while not finished:
        # This is the epsilon greedy part
        exploit = random.random() > epsilon
        
        action = 0
        currState = game.states[-1]
        stateDict = game.posDictFromNumpyArray(currState)
        agentCol = stateDict["agentCol"]
        blockRow = stateDict["blockRow"]
        blockCol = stateDict["blockCol"]
        currQs = self.qTable[blockRow][blockCol][agentCol]

        """
        Deciding our action based on either random
        selection or our q-table
        """
        if not exploit:
          # In random selection mode, we need
          # to be careful not to exit the map
          if agentCol == 0:
            action = random.choice([0, 1])
          elif agentCol == self.mapSize - 1:
            action = random.choice([-1, 0])
          else:
            action = random.choice([-1, 0, 1])
        else:
          # When exploiting, look into our q table,
          # and select the action with the highest
          # q-value
          action = np.argmax(currQs) - 1
        
        # Perform the action
        result = game.act(action)
        finished = result != 0

        # Now, update the q-table
        newQ = q_base.computeNewQVal(
            currState, action, currQs[action], LR,
            discount 
        )
        self.qTable[blockRow][blockCol][agentCol][action + 1] = newQ

      # Updating epsilon
      epsilon -= epsilonDecay
  
  """Plays the game and prints the display at each step"""
  def play(self):
    # A new game instance for this epoch
      game = Game(self.mapSize)

      # Whether or not the game is finished
      finished = False
      while not finished:        
        action = 0
        currState = game.states[-1]
        stateDict = game.posDictFromNumpyArray(currState)
        agentCol = stateDict["agentCol"]
        blockRow = stateDict["blockRow"]
        blockCol = stateDict["blockCol"]
        currQs = self.qTable[blockRow][blockCol][agentCol]

        print(currQs)

        # When exploiting, look into our q table,
        # and select the action with the highest
        # q-value
        action = np.argmax(currQs) - 1
        
        # Perform the action
        result = game.act(action)
        finished = result != 0

        print(currState)
        print(result)

agent = QTableAgent()
agent.learn(1, 0, 1000, 0.99, 1)
print(agent.qTable)
for i in range(100):
  agent.play()
  time.sleep(4)