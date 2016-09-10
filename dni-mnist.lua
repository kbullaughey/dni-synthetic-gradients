#!/usr/bin/env th

-- Train a MNIST digit classifier using DeepMind's DNI synthetic gradients
--
-- partially based on github/torch/demos/train-a-digit-classifier by Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'paths'
lapp = require 'pl.lapp'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  -f,--full                                use the full dataset
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.05)        learning rate
  -b,--batchSize     (default 10)          batch size
  -m,--momentum      (default 0)           momentum
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> trying to set ' .. opt.threads .. ' threads, got ' .. torch.getnumthreads())

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- In order to update-unlock the model, we define it as separate pieces.
-- activations - layer-1 activations
-- synthetic   - synthtic gradient prediction
-- predictions - prediction using the activations and errors fed back into the
--               synth
activations1 = nn.Sequential()
activations2 = nn.Sequential()
synthetic1 = nn.Sequential()
synthetic2 = nn.Sequential()
predictions = nn.Sequential()

------------------------------------------------------------
-- 2 hidden layers
------------------------------------------------------------
-- Activations
activations1:add(nn.Reshape(1024))
activations1:add(nn.Linear(1024, 256))
activations1:add(nn.BatchNormalization(256, nil, nil, false))
activations1:add(nn.ReLU())
activations2:add(nn.Linear(256, 256))
activations2:add(nn.BatchNormalization(256, nil, nil, false))
activations2:add(nn.ReLU())
-- Synthetic gradients for layer 1
synthetic1:add(nn.Linear(256,1024))
synthetic1:add(nn.BatchNormalization(1024, nil, nil, false))
synthetic1:add(nn.ReLU())
synthetic1:add(nn.Linear(1024,1024))
synthetic1:add(nn.BatchNormalization(1024, nil, nil, false))
synthetic1:add(nn.ReLU())
synth1Pred = nn.Linear(1024,256)
synthetic1:add(synth1Pred)
-- Synthetic gradients for layer 2
synthetic2:add(nn.Linear(256,1024))
synthetic2:add(nn.BatchNormalization(1024, nil, nil, false))
synthetic2:add(nn.ReLU())
synthetic2:add(nn.Linear(1024,1024))
synthetic2:add(nn.BatchNormalization(1024, nil, nil, false))
synthetic2:add(nn.ReLU())
synth2Pred = nn.Linear(1024,256)
synthetic2:add(synth2Pred)
-- Predictions
predictions:add(nn.Linear(256,#classes))
predictions:add(nn.LogSoftMax())
------------------------------------------------------------

-- retrieve parameters and gradients
activations1Par, activations1GradPar = activations1:getParameters()
activations2Par, activations2GradPar = activations2:getParameters()
synthetic1Par, synthetic1GradPar = synthetic1:getParameters()
synthetic2Par, synthetic2GradPar = synthetic2:getParameters()
predictionsPar, predictionsGradPar = predictions:getParameters()

-- Initialize parameters
activations1Par:uniform(-0.05, 0.05)
activations2Par:uniform(-0.05, 0.05)
synthetic1Par:uniform(-0.05, 0.05)
synth1Pred.weight:zero()
synth1Pred.bias:zero()
synthetic2Par:uniform(-0.05, 0.05)
synth2Pred.weight:zero()
synth2Pred.bias:zero()
predictionsPar:uniform(-0.05, 0.05)

----------------------------------------------------------------------
-- We use a negative log likelihood criterion for classification model
-- and a MSE criterion for synthetic gradients model.
--
classificationCriterion = nn.ClassNLLCriterion()
syntheticCriterion = nn.MSECriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
  nbTrainingPatches = 60000
  nbTestingPatches = 10000
else
  nbTrainingPatches = 2000
  nbTestingPatches = 1000
  print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

function newSGD()
  return {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    learningRateDecay = 5e-7,
    weightDecay = opt.coefL2,
  }
end

-- Create closure to evaluate f(W) and df/dW of the activations model using
-- the synthetic gradient model. Here w is the parameters of the activations
-- model.
function makeActivationsClosure(this)
  return function(w)
    -- w is probably already our parameter vector, but if not stick it in.
    if w ~= this.activationsPar then
      this.activationsPar:copy(w)
    end
    this.activationsGradPar:zero()

    -- Use the activations from the layer below to compute the activations of this layer.
    this.act = this.activations:forward(this.below.act)
    -- use the synthetic gradients model to approximate df_do
    this.syn = this.synthetic:forward(this.act)
    -- No update locking, we can immediately use our synthetic gradients to 
    -- run this module backward.
    this.activations:backward(this.below.act, this.syn)

    return this.act, this.activationsGradPar
  end
end

-- Create closure to evaluate f(W) and df/dW of the Prediction model.
-- Here w is the parameters of the prediction model.
function makePredictionsClosure(this)
  return function(w)
    -- get new parameters
    if w ~= this.predictionsPar then
      this.predictionsPar:copy(w)
    end
    this.predictionsGradPar:zero()

    -- Compute loss
    local outputs = this.predictions:forward(this.below.act)
    local f = classificationCriterion:forward(outputs, this.targets)

    -- estimate df/dW
    local df_do = classificationCriterion:backward(outputs, this.targets)
    -- Compute the actual gradient. This will be compared against the synthetic
    -- gradient to update the model that outputs the synthetic gradients.
    this.below.grad = this.predictions:backward(this.below.act, df_do):clone()

    -- update confusion
    for i = 1,opt.batchSize do
      confusion:add(outputs[i], this.targets[i])
    end

    -- return f and df/dX
    return f, this.predictionsGradPar
  end
end

-- Create closure to evaluate f(W) and df/dW of the synthetic gradients model.
-- Here w is the parameters of the synthetic gradients model.
function makeSyntheticGradientClosure(this)
  return function(w)
    -- get new parameters
    if w ~= this.syntheticPar then
      this.syntheticPar:copy(w)
    end
    this.syntheticGradPar:zero()

    -- We've already run model 'synthetic' forward, when we produced the synthetic
    -- gradients that we used to update the 'activations' model, so we can go right
    -- to the criterion.
    -- Compute a loss comparing our synthetic gradient and the real gradient.
    local synLoss = syntheticCriterion:forward(this.syn, this.grad)
    local synLossGrad = syntheticCriterion:backward(this.syn, this.grad)
    this.synthetic:backward(this.act, synLossGrad)

    -- Compute the true gradient for the layer below.
    this.activationsGradPar:zero()
    this.below.grad = this.activations:backward(this.below.act, this.grad):clone()

    -- return f and df/dX
    return synLoss, this.syntheticGradPar
  end
end

-- training function
function train(dataset)
  activations1:training()
  activations2:training()
  synthetic1:training()
  synthetic2:training()
  predictions:training()

  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,dataset:size(),opt.batchSize do
    collectgarbage()
    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end

    local layer0 = {
      act = inputs,
    }
    -- Each layer has data that will be accessible in the closure.
    local layer1 = {
      activations = activations1,
      activationsPar = activations1Par,
      activationsGradPar = activations1GradPar,
      synthetic = synthetic1,
      syntheticPar = synthetic1Par,
      syntheticGradPar = synthetic1GradPar,
      below = layer0,
    }
    local layer2 = {
      activations = activations2,
      activationsPar = activations2Par,
      activationsGradPar = activations2GradPar,
      synthetic = synthetic2,
      syntheticPar = synthetic2Par,
      syntheticGradPar = synthetic2GradPar,
      below = layer1,
    }
    local layer3 = {
      targets = targets,
      predictions = predictions,
      predictionsPar = predictionsPar,
      predictionsGradPar = predictionsGradPar,
      below = layer2,
    }

    local fEvalActivations1 = makeActivationsClosure(layer1)
    local fEvalActivations2 = makeActivationsClosure(layer2)
    local fEvalSynthetic1 = makeSyntheticGradientClosure(layer1)
    local fEvalSynthetic2 = makeSyntheticGradientClosure(layer2)
    local fEvalPredictions = makePredictionsClosure(layer3)

    sgdState1 = sgdState1 or newSGD()
    sgdState2 = sgdState2 or newSGD()
    sgdState3 = sgdState3 or newSGD()
    sgdState4 = sgdState4 or newSGD()
    sgdState5 = sgdState5 or newSGD()

    -- Perform SGD steps for each of our models:
    optim.sgd(fEvalActivations1, activations1Par, sgdState1)
    optim.sgd(fEvalActivations2, activations2Par, sgdState2)
    optim.sgd(fEvalPredictions, predictionsPar, sgdState3)
    optim.sgd(fEvalSynthetic2, synthetic2Par, sgdState4)
    optim.sgd(fEvalSynthetic1, synthetic1Par, sgdState5)
    
    -- disp progress
    xlua.progress(t, dataset:size())
  end
   
  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  local filename = paths.concat(opt.save, 'mnist.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  if paths.filep(filename) then
    os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
  end
  print('<trainer> saving network to '..filename)
  -- torch.save(filename, model)

  -- next epoch
  epoch = epoch + 1
end

-- test function
function test(dataset)
  activations1:evaluate()
  activations2:evaluate()
  predictions:evaluate()

  -- local vars
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    collectgarbage()
    -- disp progress
    xlua.progress(t, dataset:size())

    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end

    -- test samples
    local a1 = activations1:forward(inputs)
    local a2 = activations2:forward(a1)
    local preds = predictions:forward(a2)

    -- confusion:
    for i = 1,opt.batchSize do
      confusion:add(preds[i], targets[i])
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
  -- train/test
  train(trainData)
  test(testData)

  -- plot errors
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()
  end
end
