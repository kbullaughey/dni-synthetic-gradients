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
synth1 = nn.Sequential()
predictions = nn.Sequential()

------------------------------------------------------------
-- 2-layer MLP (1 hidden layer)
------------------------------------------------------------
-- Activations
activations1:add(nn.Reshape(1024))
activations1:add(nn.Linear(1024, 256))
activations1:add(nn.BatchNormalization(256, nil, nil, false))
activations1:add(nn.ReLU())
-- Synthetic gradients
synth1:add(nn.Linear(256,1024))
synth1:add(nn.BatchNormalization(1024, nil, nil, false))
synth1:add(nn.ReLU())
synth1:add(nn.Linear(1024,256))
-- Predictions
predictions:add(nn.Linear(256,#classes))
predictions:add(nn.LogSoftMax())
------------------------------------------------------------

-- retrieve parameters and gradients
activations1Par, activations1GradPar = activations1:getParameters()
synth1Par, synth1GradPar = synth1:getParameters()
predictionsPar, predictionsGradPar = predictions:getParameters()

-- Initialize parameters
activations1Par:uniform(-0.05, 0.05)
synth1Par:zero()
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
  local e = function(w)
     -- w is probably already our parameter vector, but if not stick it in.
     if w ~= this.activationsPar then
        this.activationsPar:copy(w)
     end
     this.activationsGradPar:zero()

     -- evaluate function for complete mini batch
     this.act = this.activations:forward(this.inputs)
     -- use the synthetic gradients model to approximate df_do
     this.syn = this.synthetic:forward(this.act)
     -- No update locking, we can immediately use our synthetic gradients to 
     -- run this module backward.
     this.activations:backward(this.inputs, this.syn)

     return this.act, this.activationsGradPar
  end
  return e
end

-- Create closure to evaluate f(W) and df/dW of the synthetic gradients model.
-- Here w is the parameters of the synthetic gradients model.
function makeSyntheticGradientClosure(this)
  local e = function(w)
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

     -- return f and df/dX
     return synLoss, this.syntheticGradPar
  end
  return e
end

-- training function
function train(dataset)
   activations1:training()
   synth1:training()
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

      -- Make some variables that we'll use to capture computations from the closures.
      local stage1 = {
        activations = activations1,
        activationsPar = activations1Par,
        activationsGradPar = activations1GradPar,
        synthetic = synth1,
        syntheticPar = synth1Par,
        syntheticGradPar = synth1GradPar,
        inputs = inputs,
      }
      local finalStage = stage1

      local fEvalActivations1 = makeActivationsClosure(stage1)
      local fEvalSynthetic1 = makeSyntheticGradientClosure(stage1)

      -- Create closure to evaluate f(W) and df/dW of the Prediction model.
      -- Here w is the parameters of the prediction model.
      local fEvalPredictions = function(w)
         -- get new parameters
         if w ~= predictionsPar then
            predictionsPar:copy(w)
         end
         predictionsGradPar:zero()

         -- Compute loss
         local outputs = predictions:forward(finalStage.act)
         local f = classificationCriterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = classificationCriterion:backward(outputs, targets)
         -- Compute the actual gradient. This will be compared against the synthetic
         -- gradient to update the model that outputs the synthetic gradients.
         finalStage.grad = predictions:backward(finalStage.act, df_do)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,predictionsGradPar
      end

       sgdState1 = sgdState1 or newSGD()
       sgdState2 = sgdState2 or newSGD()
       sgdState3 = sgdState3 or newSGD()

       -- Perform SGD steps for each of our models:
       optim.sgd(fEvalActivations1, activations1Par, sgdState1)
       optim.sgd(fEvalPredictions, predictionsPar, sgdState2)
       optim.sgd(fEvalSynthetic1, synth1Par, sgdState3)
    
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
   synth1:evaluate()
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
      local a = activations1:forward(inputs)
      local preds = predictions:forward(a)

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
