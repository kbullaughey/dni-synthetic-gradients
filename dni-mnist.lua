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
  -c,--condition                           condition synthetic gradients on labels
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

-- Layer sizes. 
l0H = 1024
l1H = 256
l2H = 256
s1H = 1024
s2H = 1024

-- BatchNormalization parameters
bnMomentum = 0.25
------------------------------------------------------------
-- 2 hidden layers
------------------------------------------------------------
-- Activations for layer 1
activations1:add(nn.Reshape(l0H))
activations1:add(nn.Linear(l0H, l1H))
activations1:add(nn.BatchNormalization(l1H, nil, bnMomentum, false))
activations1:add(nn.ReLU())
-- Activations for layer 2
activations2:add(nn.Linear(l1H, l2H))
activations2:add(nn.BatchNormalization(l2H, nil, bnMomentum, false))
activations2:add(nn.ReLU())

-- If using the conditioning on labels (cDNI) then we tack these
-- on to the activations. When conditioning we model the gradient with
-- a simple linear transform (0-layer neural net). When not conditioning
-- on labels, we use a much more capable, 2-layer neural net.
if opt.condition then
  print("conditioning on labels (i.e., cDNI)")
  s1In = l1H + #classes
  s2In = l2H + #classes
  -- Synthetic gradients for layer 1, activations joined with labels
  synthetic1:add(nn.JoinTable(1,1))
  synth1Pred = nn.Linear(l1H+#classes,l1H)
  -- Synthetic gradients for layer 2, activations joined with labels
  synthetic2:add(nn.JoinTable(1,1))
  synth2Pred = nn.Linear(l2H+#classes,l2H)
else
  -- Synthetic gradients for layer 1
  synthetic1:add(nn.Linear(l1H,s1H))
  synthetic1:add(nn.BatchNormalization(s1H, nil, bnMomentum, false))
  synthetic1:add(nn.ReLU())
  synthetic1:add(nn.Linear(s1H,s1H))
  synthetic1:add(nn.BatchNormalization(s1H, nil, bnMomentum, false))
  synthetic1:add(nn.ReLU())
  synth1Pred = nn.Linear(s1H,l1H)
  -- Synthetic gradients for layer 2
  synthetic2:add(nn.Linear(l2H,s2H))
  synthetic2:add(nn.BatchNormalization(s2H, nil, bnMomentum, false))
  synthetic2:add(nn.ReLU())
  synthetic2:add(nn.Linear(s2H,s2H))
  synthetic2:add(nn.BatchNormalization(s2H, nil, bnMomentum, false))
  synthetic2:add(nn.ReLU())
  synth2Pred = nn.Linear(s2H,l2H)
end

synthetic1:add(synth1Pred)
synthetic2:add(synth2Pred)

-- Predictions
predictions:add(nn.Linear(l2H,#classes))
predictions:add(nn.LogSoftMax())
------------------------------------------------------------

-- retrieve parameters and gradients
activations1Par, activations1GradPar = activations1:getParameters()
activations2Par, activations2GradPar = activations2:getParameters()
synthetic1Par, synthetic1GradPar = synthetic1:getParameters()
synthetic2Par, synthetic2GradPar = synthetic2:getParameters()
predictionsPar, predictionsGradPar = predictions:getParameters()

-- Initialize parameters
r = 0.07
activations1Par:uniform(-r, r)
activations2Par:uniform(-r, r)
synthetic1Par:uniform(-r, r)
synth1Pred.weight:zero()
synth1Pred.bias:zero()
synthetic2Par:uniform(-r, r)
synth2Pred.weight:zero()
synth2Pred.bias:zero()
predictionsPar:uniform(-r, r)

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
--    learningRateDecay = 5e-7,
    weightDecay = opt.coefL2,
  }
end

function newAdam()
  return {
    learningRate = opt.learningRate,
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
    if opt.condition then
      this.synGrad = this.synthetic:forward({this.act,this.labels})
    else
      this.synGrad = this.synthetic:forward(this.act)
    end
    -- No update locking, we can immediately use our synthetic gradients to 
    -- run this module backward.
    this.below.bpGrad = this.activations:backward(this.below.act, this.synGrad)

    return this.act, this.activationsGradPar
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
    local synLoss = syntheticCriterion:forward(this.synGrad, this.bpGrad)
    local synLossGrad = syntheticCriterion:backward(this.synGrad, this.bpGrad)
    if opt.condition then
      this.synthetic:backward({this.act,this.labels}, synLossGrad)
    else
      this.synthetic:backward(this.act, synLossGrad)
    end

    -- return f and df/dW
    return synLoss, this.syntheticGradPar
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
    this.below.bpGrad = this.predictions:backward(this.below.act, df_do)

    -- update confusion
    for i = 1,opt.batchSize do
      confusion:add(outputs[i], this.targets[i])
    end

    -- return f and df/dW
    return f, this.predictionsGradPar
  end
end

-- training function
function train(dataset)

  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local perm = torch.randperm(dataset:size())
  for t = 1,dataset:size(),opt.batchSize do
    collectgarbage()
    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[perm[i]]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end

    local batchLabels
    if opt.condition then
      batchLabels = torch.zeros(opt.batchSize, #classes)
      for i=1,opt.batchSize do
        batchLabels[i][targets[i]] = 1
      end
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

    -- We provide the labels to the synthetic gradient modules if using cDNI.
    if opt.condition then
      layer1.labels = batchLabels
      layer2.labels = batchLabels
    end

    local fEvalActivations1 = makeActivationsClosure(layer1)
    local fEvalActivations2 = makeActivationsClosure(layer2)
    local fEvalSynthetic1 = makeSyntheticGradientClosure(layer1)
    local fEvalSynthetic2 = makeSyntheticGradientClosure(layer2)
    local fEvalPredictions = makePredictionsClosure(layer3)

    --local optimizer = optim.sgd
    --local stateFactory = newSGD
    local optimizer = optim.adam
    local stateFactory = newAdam

    optimState1 = optimState1 or stateFactory()
    optimState2 = optimState2 or stateFactory()
    optimState3 = optimState3 or stateFactory()
    optimState4 = optimState4 or stateFactory()
    optimState5 = optimState5 or stateFactory()

    -- Notation matching Figure 2 in DNI paper
    -- update f_{i}
    optimizer(fEvalActivations1, activations1Par, optimState1)
    -- update f_{i+1}
    optimizer(fEvalActivations2, activations2Par, optimState2)
    -- update M_{i+1}
    optimizer(fEvalSynthetic1, synthetic1Par, optimState3)
    -- update f_{i+2}
    optimizer(fEvalPredictions, predictionsPar, optimState4)
    -- update M_{i+1}
    optimizer(fEvalSynthetic2, synthetic2Par, optimState5)
    
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
