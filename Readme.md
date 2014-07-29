StocPy
====

StocPy is an expressive [probabilistic programming language](http://probabilistic-programming.org) written in Python. The language follows the "lightweight implementations" style introduced by Wingate, Stuhlmüller and Goodman ([link to pdf](http://www.mit.edu/~ast/papers/lightweight-mcmc-aistats2011.pdf)).

Features
---

* Intuitive, succinct and flexible model specification , thanks to Python
* Easily extensible to handling different probability distributions
* Modular inference engine architecture allowing for the usage of different inference techniques. Currently Metropolis, Slice Sampling and combinations of the two are supported (note, slice sampling is still work in progress and may not work correctly on all models).

Basic Usage
---
A model is specified as a normal Python function which uses the probabilistic primitives provided by the StocPy library. For instance, inferring the mean of a normal distribution of variance 1 based on a single data point would be written as:

    def guessMean():
       mean = stocPy.normal(0, 1, obs=True)
       stocPy.normal(mean, 1, cond=2)

Here, we define a prior on the mean as a Normal(0,1) and also say that we wish to observe what values the mean will take in our simulation (i.e. sample the mean). In the next line we condition the model on our only data point (2 in this case).

Once this model is defined we can perform inference on it by calling:

    samples = stocPy.getSamples(guessMean, 10000, alg="met")
Where we are asking for 10,000 samples generated from the model with the Metropolis inference technique (which is also the default technique).

Finally, several utility functions are provided. For instance, to quickly visualise the results of our inference, we could call:

    stocPy.plotSamples(samples)

For more usage examples (including more advanced cases), please see the provided models.


Installation
---
The easiest way is to clone the repository with `git clone git@github.com:RazvanRanca/StocPy.git` and install StocPy using `python setup.py install`  (which might require root access). 

The repository consists of a master branch and a development branch. The later has some work in progress features which are not yet fully functional.


Contact
---
Please send any questions/suggestions/issues to:

Razvan Ranca - ranca.razvan@gmail.com
