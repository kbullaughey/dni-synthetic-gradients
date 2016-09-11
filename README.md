# Torch implementation of Decoupled Neural Interfaces

Here I reproduce some of the MNIST experiments from DeepMind's paper, [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343).

My starting point was the MNIST torch demo, [train a digit classifer](https://github.com/torch/demos/tree/master/train-a-digit-classifier).

## Initial impressions on implementing DNI.

Decoupled neural interfaces turns out to be incredibly simple to implement, particularly in torch. 

To review, the normal forward/backpropagation training for a feed-forward neural nets can be done in a single SGD update step:

1. Update 1:
    1. Evaluate the whole net through to predictions as one function, f.
    2. Evaluate the loss with respect to targets.
    3. Backpropagate through the criterion to get the gradient of the error wrt the predictions.
    4. Update the parameters by backpropagating 

In torch, making the actual update looks like this:

    optim.sgd(fEval, parameters, state)

And the `feval` function has forward/backward steps corresponding to the above 4 steps that look like this:

    outputs = model:forward(inputs)
    f = criterion:forward(outputs, targets)
    df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do)

For decoupled neural interfaces, we can perform the updates in an unlocked fashion as soon as the (synthetic) gradient becomes available. One way this can be done is with 5 updates of the optimizer, each working on a smaller piece of the model (i.e., one layer or one synthetic gradient model).

Thus perform 5 updates to parameters each minibatch. Each update is accomplished with a call to `optim.adam(f, par, state)`. The following notation corresponds to Figure 2 in the DNI paper but I use ^δ to refer to the synthetic gradient estimate of δ.

1. Update 1:
    1. Evaluate f<sub>i</sub>
    2. Evaluate M<sub>i+1</sub> to produce ^δ<sub>i</sub>.
    3. Update f<sub>i</sub> by backpropagating ^δ<sub>i</sub>.
2. Update 2:
    1. Evaluate f<sub>i+1</sub>
    2. Evaluate M<sub>i+2</sub> to produce ^δ<sub>i+1</sub>.
    3. Update f<sub>i+1</sub> by backpropagating ^δ<sub>i+1</sub>.
3. Update 3:
    1. Evaluate the loss ‖^δ<sub>i</sub> - δ<sub>i</sub>‖. Notice that δ<sub>i</sub> is the result of backpropagating ^δ<sub>i+1</sub> through f<sub>i+1</sub> (computed in step 2.3). This is not the true gradient, as we haven't compared our predictions to the targets yet.
    1. Update M<sub>i+1</sub>.
4. Update 4:
    1. Evaluate f<sub>i+2</sub>, which in our case is our predictions.
    2. Compute the classification loss comparing our predictions to the targets.
    3. Update f<sub>i+2</sub> by backpropagating the classification loss back through the prediction layer.
5. Update 5:
    1. Evaluate the loss ‖^δ<sub>i+1</sub> - δ<sub>i+1</sub>‖. Here δ<sub>i+1</sub> is the actual backpropagated loss from the prediction. But if we had more layers, this could also be a backpropagated synthetic gradient (as in step 3.1).
    2. Update M<sub>i+1</sub>.

This progression illustrates the update-decoupling. The bulk of the updates are performed before the actual loss is computed (in step 4.2).

In torch code, this involves 5 updates using our optimizer:

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

If you're interested in the details of the 5 eval functions, see the script `dni-mnist.lua`. Naturally we'd want to handle the layers in a loop to make it work to arbitrary depth, but I've implemented each separately for pedagogical purposes.

## Data

The MNIST data I use are from torch on AWS:

[https://s3.amazonaws.com/torch7/data/mnist.t7.tgz](https://s3.amazonaws.com/torch7/data/mnist.t7.tgz))

These are 32x32 images. All the feed-forward models treat this as a 1024-length vector.

## Baselines

The following two baselines use regular backpropagation for estimating the gradient.

### Stock demo

The script `mnist-simple.lua` is basically the original `train-on-mnist.lua` demo script but stripped down to include only the MLP and SGD (I've stripped out the convolutional net and logistic regression, and LBFGS optimization).

It uses a MLP with one hidden layer of size 2048, a Tanh non-linearity, and a linear projection down to the 10 classes, using a LogSoftMax output with a negative log-likelihood loss function. Training is done using regular back-propagation for estimating gradients and SGD for optimization.

If we run it with a batch size of 250 and the default learning rate (0.05):

    ./mnist-simple.lua -f -b 250

We can get a training error of 2.0% by epoch 46.

### BackProp baseline from paper

The script `mnist-relu.lua` matches the simplest backpropagation fully-connected network (FCN). The baseline reported here is closest to the model used in 3-layer FCN Bprop model reported in the first row, second column of Table 1. The only difference is that I have used SGD instead of Adam for optimization.

Otherwise the architecture is the same, featuring two hidden layers (size 256) comprising a Linear transform, batch normalization, and then a rectified linear unit (ReLU). Then there is a projection layer down to the 10 classes, with a LogSoftMax and negative log-likelihood loss, as above.

If we run it with a batch size of 250 and the default learning rate (0.05):

    ./mnist-relu.lua -f -b 250

We can get a training error of 2.0% by epoch 21.

## DNI implementation details

I have tried to stick as close as possible to the architecture described in the paper.

### DNI model

The script `dni-mnist.lua` uses synthetic gradient estimates after each hidden layer to remove the update-lock that is usually associated with backpropagation. Given there are two hidden layers in these experiments, there are two synthetic gradients updated.

This model involves two hidden layers each with 256 units (a Linear map, batch normalization, and ReLU transform, as above).

For the synthetic gradients, I follow the paper and use a neural network with two hidden layers each with 1024 units (a Linear map, batch normalization, and ReLU transform), followed by a linear map to get back to the size of the gradient, 256.

Using a batch size of 250 and a learning rate of 0.0001:

    ./dni-mnist.lua -b 250 -f -r 0.0001

I only managed to reach an error rate of 2.8% after 249 epochs (or 60k iterations) and even by 770 epochs (185k iterations) it still hadn't gotten below 2.7% error.

The learning rate above (0.0001) is 3x the rate reported in the paper. But decreasing it didn't seem to help. It's worth noting that I was able to use a learning rate 10x higher yet (0.001) when conditioning on the labels (cDNI model). Such a high learning rate trained poorly for the unconditional model here. This probably relates to the very low amount of information in the synthetic gradients when not conditioning on the labels. My theory is that the unconditional synthetic gradient model is tasked with making both a rough prediction of the class as well as modeling how the activations should be updated given this prediction. This seems like a lot to expect from the synthetic gradient neural net.

### cDNI model

The script `dni-mnist.lua` when passed the `-c` parameter conditions the synthetic gradient estimates on the labels. It is identical to the DNI model except for how the synthetic gradients are computed. 
Thus, in addition to the activations (or inputs) from the layer below, the synthetic gradient module also takes as input the labels. 

I follow the suggestion in the paper that a simple linear transform was all that is needed to estimate the gradients. In practice this entails joining the activations and the labels, using `nn.JoinTable(1,1)`, and then having a simple linear map, using `nn.Linear(256+10,256)`. This astonishingly simple gradient estimate seems to do the trick. 

This result is closest to the result in the 3-layer FCN cDNI model reported in the first row, fourth column of Table 1 in the paper.

If we run with a batch size of 250 and a learning rate of 0.001:

    ./dni-mnist.lua -b 250 -f -r 0.001 -c

We get an error rate of 2.0% by epoch 80. I believe this corresponds to 19k iterations. This seems to be converging somewhat slower than the equivalent cDNI model in the paper (red line in figure next to Table 1).

## Remarks

0. The synthetic gradients seem to act as a strong regularizer, which seems a good thing.
0. For simple feed-forward models like those in these experiments, there is really no point of using synthetic gradients, nor it this their intended purpose. These demos are just to illustrate how they are implemented.
0. Synthetic gradients seem to open up a huge world of elaborate architectures composed of asynchronous, decoupled subsystems. That they can be decoupled seems to make such subsystems much more easily composable. It will be interesting to see where this path leads.
0. My guess as to why synthetic gradients conditioning on labels (cDNI) are so good at learning deep nets with many layers (up to 21 layers as reported in the paper) is that it has more to do with conditioning on the labels than the fact they're using synthetic gradients. Conditioning on the gradients probably is acting like skip connections or something.

## Notes

0. I use a batch size of 250 instead of 256, as was used in the DNI paper, because torch gets confused between the batch dimension and the data dimension when both are 256 and I didn't want bother fixing it (which I'm sure is possible by passing some extra parameters somewhere).
