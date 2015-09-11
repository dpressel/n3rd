n3rd
====

## Simple, no frills, easy to use (and understand) neural nets / deep learning in Java

I wanted a simple, easy to use, flexible (enough) neural net architecture for experiments, and at the same time, I wanted to see how far we can push Leon Bottou's ubiquitous SGD approach as a backbone for deep neural nets.  What I ended up is very simple, and does reuse a lot of the basic architecture (in the form of [sgdtk](https://github.com/dpressel/sgdtk/blob/master/README.md), my modular implementation of SGD for structured and unstructured predictions based on Bottou's sample code), keeping the actual code very minimal. The code also draws from various other sources of inspiration for its own contributions, particularly Torch.  Due to how the SGD framework is structured (primarily for linear classification problems), we are left handling backprop in the model, outside of the actual Learner, a necessary by-product (I think) of preserving the original structure.

For the time being, this code is implemented in pure CPU Java, though I believe other backends should be possible well (I hope to address this in the near future).  Adagrad is currently used to train the network layers.
