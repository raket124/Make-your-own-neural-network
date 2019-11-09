# Make your own neural network

This repository is inspired by the book [Make your own neural network](https://www.goodreads.com/book/show/29746976-make-your-own-neural-network).
It takes the well known [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwriten numbers (60.000 training and 10.000 test samples) 
and uses them to train and validate a fully connected feed forward neural network.
The experiments show the effects of the paramters used to train the neural network.

The goals for this repository are: 
- Reproduction of the experiments
- Command line only (except for coding)

## Tools & Libraries
- Vagrant (Virtual machine provider)
- Python (Programming language)
  - Numpy (Math library)
  - Matplotlib (Data visualization library)
- FFmpeg + ImageMagic (Encoders)
- Atom (IDE)

## Experiments

### Experiment 1 - A training run with fixed values.
The first experiment is a single training run with fixed values.
By using generic parameters the neural network is able to achieve a decent score of XXXX.



### Experiment 2 - The influence of the learning rate
The second experiment trains the neural network by using different learning rates.
The expected behaviour is that a small learning rate will prevent the neural network from achieving a proper generic answer.
While a large learning rate will cause the neural network to continuously overshoot the correct answer.

![lr]


### Experiment 3 - Training multiple rounds

![ep]

### Experiment 4 - Learning rates vs Epochs

![lrep]

### Experiment 5 - Inverse query (estimation)


[lr]: https://github.com/raket124/Make-your-own-neural-network/blob/master/Code/Output/LearningRate.png "Learning rate plot"
[ep]: https://github.com/raket124/Make-your-own-neural-network/blob/master/Code/Output/Epoch.png "Epoch plot"
[lrep]: https://github.com/raket124/Make-your-own-neural-network/blob/master/Code/Output/EpochAndLearningRate.gif "Epoch vs learning rate plot"





