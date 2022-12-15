"""An implementation of DQN; basically using 2 neural nets in alternating
training to approximate the Q function, learning off-policy with the Bellman
Equation"""

import random
from collections import Counter

import torch
from torch import nn

import environment
import q_base

class DQNetwork(nn.Module):
  def __init__(self, mapSize):
    self.mapSize = mapSize
    super(DQNetwork, self).__init__()

    NUM_CONV_LAYERS = 4

    # Constructing the CNN layers on their own first,
    # then turning them into a module
    cnLayers = [
        nn.Conv2d(2, 8, kernel_size=2)
    ]

    for i in range(NUM_CONV_LAYERS - 1):
      cnLayers.append(
        nn.Conv2d(8, 8, kernel_size=2)
      )

    cnBlock = nn.Sequential( *cnLayers )

    # Now that our CNN block is constructed, determine the layer size
    # of the first dense layer after the CNN block by passing a tensor
    # through it and seeing the shape at the end
    cnnOutTensor = cnBlock( torch.zeros( 2, mapSize + 2, mapSize ) )
    cnnOutLength = torch.flatten(cnnOutTensor).shape[0]

    # Now that we know the size, make our dense stack
    denseStack = nn.Sequential(
        nn.Flatten(start_dim=0),
        nn.Linear(cnnOutLength, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 30),
        nn.LeakyReLU(),
        nn.Linear(30, 3),
        nn.Tanh()
    )

    # Now make our full network
    self.fullNetwork = nn.Sequential(
        cnBlock,
        denseStack
    )
  
  def forward(self, X):
    return self.fullNetwork(X)

class DQNAgent:
  # Copies the predictor model into the target model
  def __updateTarget(self):
    self.target.load_state_dict(self.predictor.state_dict())

  def __init__(self, mapSize):
    self.mapSize = mapSize

    """
    Below are the networks used for this system. They are named as follows:
    Predictor: The predictor network is the one that's actually responsible for
    making the quality predictions during training, and its Q values are used to select
    the best action when in exploit mode. As well, its Q-values are the ones most frequently
    updated against the target Q valeus produced by the Bellman Equation.

    Target: The target network is used to actually compute the result of the Bellman Equation,
    resulting in a New Q value for a given state-action pair. This new pair is then added to the
    experience replay buffer, from which the predictor learns. This target network is updated
    periodically but always lags behind the predictor, helping with stability.
    """

    self.predictor = DQNetwork(self.mapSize)
    self.target = DQNetwork(self.mapSize)

  """
  Trains the DQN agent. Each epoch corresponds to one update of the target network, while
  the experiences per epoch indicates how many full games are played in each epoch.
  """
  def train(self, epochs, experiencesPerEpoch, epsilon, epsilonDecayPerEpoch):
    "Defining all hyperparamaters and prepping for training"
    LR = 5e-4
    OPTIM = torch.optim.Adam(self.predictor.parameters(), lr=LR)
    # Batch size is how many Bellman Updates are used per
    # mini-batch while updating the predictor
    BS = 64
    # Max # of experiences to play through before training the
    # predictor network on all data in the experience replay
    # buffer
    MAX_EXPERIENCES_PER_UPDATE = 200
    # Number of times to re-use the data in the replay buffer.
    # Makes training more efficient by needing less evaluations by the
    # target net to produce belmman update.
    DATA_REUSE_FACTOR = 2

    for EPOCH in range(epochs):
      # The first thing we need to do in each epoch is fill our replay buffer
      # with the number of experiences between each update, and then update. And keep
      # repeating till we've hit the max number of experiences per epoch
      
      # This variable records how many experiences we've seen in this epoch
      # so far
      previousExperiences = 0

      while previousExperiences < experiencesPerEpoch:
        # Initializing a replay buffer, and generating training samples.
        # Stores a sequence of tuples of the following format:
        # (depth 2 pytorch state array, action(zero indexed, not the actual action), qValue) 
        replayBuffer = []
        for experienceNumber in range( 
          min(MAX_EXPERIENCES_PER_UPDATE, experiencesPerEpoch - previousExperiences) ):
          # Generating a single training sample and adding to our replay buffer

          game = environment.Game(self.mapSize)

          # Whether or not the game is finished
          finished = False
          while not finished:
            # The action we're deciding on for this timestep
            action = 0
            currState = game.states[-1]
            stateDict = game.posDictFromNumpyArray(currState)
            agentCol = stateDict["agentCol"]
            blockRow = stateDict["blockRow"]
            blockCol = stateDict["blockCol"]

            # Getting the depth 2 array that corresponds with the current
            # state
            depthTwoArray = environment.Game._depthTwoFromPosStates(
                  blockCol, blockRow, agentCol, self.mapSize)
                
            # Converting it into a tensor that can be used by pytorch
            pytorchDepthTwoArray = torch.tensor(depthTwoArray, dtype=torch.float32)

            # We need the target Qs in this case, since they are the
            # Qs used to produce the updates for the predictor
            targetQs = torch.zeros( (1) )
            with torch.no_grad():
              targetQs = self.target(pytorchDepthTwoArray)

            # Exploration
            if random.random() <= epsilon:
              action = random.randint(-1, 1)
            # Exploitation
            else:
              # When exploiting, use the prediction
              # network to output qs, and then pick action
              # with the highest q
              with torch.no_grad():
                predictorQs = self.predictor(pytorchDepthTwoArray)
                action = torch.argmax(predictorQs) - 1
            
            # Now, act
            result = game.act(action)

            # Update finished to true if applicable
            finished = result != 0

            # Now that we've acted, compute an updated Q value
            # for that state action pair, using the old Qs for the
            # update.
            newQ = q_base.computeNewQVal(
              currState, action, targetQs[action + 1], 1e-1, 0.99
            )

            # Add the newQ as a training sample to the replay
            # buffer
            replayBuffer.append(
              ( pytorchDepthTwoArray, action + 1, newQ )
            )

            # Increment experience count
            previousExperiences += 1

        # Now, learn from the previous experiences, repeating
        # based on the data reuse factor
        for reuse in range(DATA_REUSE_FACTOR):
          # First, shuffle the data
          random.shuffle(replayBuffer)

          # Now, iterate over the data, pulling a full sized batch at each
          # run. If we're towards the end of the list, just randoly sample
          # from the buffer to get enough samples to make a full batch
          for batchStartIndex in range(0, len(replayBuffer), BS):
            # This list stores all the errors for this mini-batch
            miniBatchErrors = []

            batchEndIndex = min( len(replayBuffer) - 1, batchStartIndex + BS )
            trainingTuples = replayBuffer[ batchStartIndex : batchEndIndex + 1 ]

            # Randomly sample more samples if we're missing samples to make a full
            # batch
            if len(trainingTuples) < BS:
              trainingTuples.extend( random.sample(replayBuffer, k = BS - len(trainingTuples)) )
            
            # At this point, we have a full training batch. Pass this through our model, compute
            # the error, and then store it
            for depthTwoState, actionIndex, expectedQ in trainingTuples:
              currentPredictQs = self.predictor(depthTwoState)
              qOfAction = currentPredictQs[actionIndex]
              mseError = ( qOfAction - expectedQ ) ** 2

              miniBatchErrors.append(mseError)
            
            # Now, average the error over the minibatch, and then back-propogate
            miniBatchErrorStacked = torch.vstack(miniBatchErrors)
            miniBatchMeanError = torch.mean(miniBatchErrorStacked)

            OPTIM.zero_grad()
            miniBatchMeanError.backward()

            # Clipping the gradient
            # torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 2)
            OPTIM.step()

            # Print error
            print(miniBatchMeanError)

      # Now, update the target net
      self.__updateTarget()

      # Now epsilon decay
      epsilon -= epsilonDecayPerEpoch
  
  """
  Actually plays games using the agent
  """
  def play(self, numGames):
    # Store results
    results = []

    with torch.no_grad():
      for i in range(numGames):
        game = environment.Game(self.mapSize)
        
        finished = False
        while not finished:
          currState = game.states[-1]
          posDict = environment.Game.posDictFromNumpyArray(currState)
          depthTwoNumpy = environment.Game._depthTwoFromPosStates(
            posDict["blockCol"], posDict["blockRow"], posDict["agentCol"], self.mapSize)
          pytorchDepthTwo = torch.tensor(depthTwoNumpy, dtype=torch.float32)

          targetQs = self.target(pytorchDepthTwo)
          action = torch.argmax(targetQs) - 1

          result = game.act(action)
          if result != 0:
            results.append(result)
            finished = True
    return results

myDQN = DQNAgent(10)

numEpochs = 100
epsilonDecay = 1 / (numEpochs * .45)
myDQN.train( numEpochs, 3000, 1, epsilonDecay )

results = myDQN.play(1000)
resultCounter = Counter(results)
print(resultCounter)