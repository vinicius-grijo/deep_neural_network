# deep_neural_network

Neural Networks as Approximators and the Initialization Dilemma

Among the various purposes of a Deep Neural Network, there is also that of approximating a given function.

Taking the values of cos(2* \pi * x ) and random variations obtained from a normal distribution of mean $0$ and variance $0.5^{2}$ as a data set, it is possible to train a Deep Neural Network.

The objective of this Network is to approximate the cosine function, trying to ignore the noise added to the values.

A key point of this project is the following: if the optimization process is started with its "default" parameters, the result is not satisfactory. The network fitted in this procedure is called "dumb". On the other hand, if the process starts with values such that the variance of the network nodes is constant throughout its layers, then the process presents pleasant results. This network that presents better results is called "smart".

The adjusted networks are both formed by a deep layer with $12$ neurons and uses the Hyperbolic Tangent as Activation Function, while the optimizer used was "adam", an extension of the classic Stochastic Gradient Descent.

The results can be seen in the attached figure.
