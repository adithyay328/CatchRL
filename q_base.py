# Implements some basic utils and functions needed
# for Q-learning, irresepctive of implementation
# via table or deep network
import environment

# Takes in a map's state array, and
# returns the value of that position
def value_function(stateArray):
    # The reward word an action is basically
    # how much closer it gets us to the target column
    posDict = environment.Game.posDictFromNumpyArray(stateArray)
    blockCol = posDict["blockCol"]
    agentCol = posDict["agentCol"]
    mapWidth = stateArray.shape[1]
    
    # This is the reward : max width - distance from that
    return ( mapWidth - abs(blockCol - agentCol) )

# Given a map state and an action, compute the
# q-val of that action. We compute that as
# the value of the inital state subtracted from
# the value of the new state
def quality_function(stateArray, action):
    oldValue = value_function(stateArray)

    posDict = environment.Game.posDictFromNumpyArray(stateArray)
    blockCol = posDict["blockCol"]
    blockRow = posDict["blockRow"]
    agentCol = posDict["agentCol"]
    mapWidth = stateArray.shape[1]

    newState = environment.Game._numpyFromPosStates(
        blockCol, blockRow, agentCol + action, mapWidth
    )

    newValue = value_function(newState)

    return newValue - oldValue


# Computes the new q value for a state-action
# pair, using the Bellman Equation
def computeNewQVal(stateArray, action, currentQ, learningRate, discountFactor):
  posDict = environment.Game.posDictFromNumpyArray(stateArray)
  blockRow = posDict["blockRow"]
  blockCol = posDict["blockCol"]
  agentCol = posDict["agentCol"]

  # Computing the new board position
  newBlockRow = blockRow + 1
  newAgentCol = agentCol + action

  # One case we need to worry about is if this is an invalid
  # move; in that case, return -1
  NEG = -1
  if newAgentCol < 0 or newAgentCol >= stateArray.shape[1]:
    return NEG
  
  # The new board array
  newPosArray = environment.Game._numpyFromPosStates(blockCol, newBlockRow, 
      newAgentCol, stateArray.shape[1])
  
  # Computing quality of that action
  actionReward = quality_function(stateArray, action)

  # Compute max reward value from this new state
  possibleNextRewards = []
  for action in [-1, 0, 1]:
    # Make sure action is valid
    if newAgentCol == 0 and action == -1:
        continue
    elif newAgentCol == stateArray.shape[1] - 1 and action == 1:
        continue

    possibleNextRewards.append(quality_function(newPosArray, action))
  maxFutureReward = max(possibleNextRewards)

  # Return our result
  result = currentQ + learningRate * (  actionReward +  discountFactor * maxFutureReward - currentQ)
  
  return result