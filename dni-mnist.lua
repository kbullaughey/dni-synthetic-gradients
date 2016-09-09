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
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -r,--learningRate  (default 0.05)        learning rate
   -b,--batchSize     (default 10)          batch size
   -M,--momentum      (default 0)           momentum
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

if opt.network == '' then
   -- In order to update-unlock the model, we define it as separate pieces.
   -- activations - layer-1 activations
   -- synthetic   - synthtic gradient prediction
   -- predictions - prediction using the activations and errors fed back into the
   --               synth
   activations = nn.Sequential()
   synthetic = nn.Sequential()
   predictions = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*3*3))
      model:add(nn.Linear(64*3*3, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------
   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- 2-layer MLP (1 hidden layer)
      ------------------------------------------------------------
      -- Activations
      activations:add(nn.Reshape(1024))
      activations:add(nn.Linear(1024, 1024))
      activations:add(nn.Tanh())
      -- Synthetic gradients
      synthetic:add(nn.Linear(1024,1024))
      -- Predictions
      predictions:add(nn.Linear(1024,#classes))
      predictions:add(nn.LogSoftMax())
      ------------------------------------------------------------
   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
activationsPar, activationsGradPar = activations:getParameters()
syntheticPar, syntheticGradPar = synthetic:getParameters()
predictionsPar, predictionsGradPar = predictions:getParameters()

-- Initialize parameters
activationsPar:uniform(-0.05, 0.05)
syntheticPar:zero()
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

-- training function
function train(dataset)
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

      -- Make some variables that we'll use to capture computations from the
      -- closures.
      local act -- activations
      local syn -- synthetic gradients
      local grad -- actual gradients

      -- Create closure to evaluate f(W) and df/dW of the activations model using
      -- the synthetic gradient model. Here w is the parameters of the activations
      -- model.
      local fevalActivations = function(w)
         -- w is probably already our parameter vector, but if not stick it in.
         if w ~= activationsPar then
            activationsPar:copy(w)
         end
         activationsGradPar:zero()

         -- evaluate function for complete mini batch
         act = activations:forward(inputs)
         -- use the synthetic gradients model to approximate df_do
         syn = synthetic:forward(act)
         -- No update locking, we can immediately use our synthetic gradients to 
         -- run this module backward.
         activations:backward(inputs, syn)

         return act, activationsGradPar
      end

      -- Create closure to evaluate f(W) and df/dW of the Prediction model.
      -- Here w is the parameters of the prediction model.
      local fevalPredictions = function(w)
         -- get new parameters
         if w ~= predictionsPar then
            predictionsPar:copy(w)
         end
         predictionsGradPar:zero()

         -- Compute loss
         local outputs = predictions:forward(act)
         local f = classificationCriterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = classificationCriterion:backward(outputs, targets)
         -- Compute the actual gradient. This will be compared against the synthetic
         -- gradient to update the model that outputs the synthetic gradients.
         grad = predictions:backward(act, df_do)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,predictionsGradPar
      end

      -- Create closure to evaluate f(W) and df/dW of the synthetic gradients model.
      -- Here w is the parameters of the synthetic gradients model.
      local fevalSynthetic = function(w)
         -- get new parameters
         if w ~= syntheticPar then
            syntheticPar:copy(w)
         end
         syntheticGradPar:zero()

         -- We've already run model 'synthetic' forward, when we produced the synthetic
         -- gradients that we used to update the 'activations' model, so we can go right
         -- to the criterion.
         -- Compute a loss comparing our synthetic gradient and the real gradient.
         local synLoss = syntheticCriterion:forward(syn, grad)
         local synLossGrad = syntheticCriterion:backward(syn, grad)
         synthetic:backward(act, synLossGrad)

         -- return f and df/dX
         return synLoss,syntheticGradPar
      end

       -- Perform SGD steps for each of our models:
       sgdState = sgdState or {
          learningRate = opt.learningRate,
          momentum = opt.momentum,
          learningRateDecay = 5e-7,
          weightDecay = opt.coefL2,
       }
       optim.sgd(fevalActivations, activationsPar, sgdState)
       optim.sgd(fevalPredictions, predictionsPar, sgdState)
       optim.sgd(fevalSynthetic, syntheticPar, sgdState)
    
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
      local a = activations:forward(inputs)
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
