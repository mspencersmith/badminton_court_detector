# Badminton Court Detector

In this project I built and trained a convolutional neural network (Pytorch model) to detect badminton courts in images.

## Technologies
### Python 3.6
* Pytorch
* Matplotlib
* OpenCV
* Numpy

## Building the Data Set

In order to ensure the highest level of accuracy from the model it was important to gather a large quantity of images, while at the same time ensuring that the data set was balanced.

I was able to collect 12,000 images of badminton courts by utilising OpenCV's VideoCapture() and read() methods on videos of badminton matches. As a former professional player I had many videos to choose from, along with other videos and stock images. I made sure I captured as many different arenas and angles as possible, and also flipped images to increase the number I had to work with.

For balance I also collected 12,000 random stock images to test and train the model with, ensuring again to gather a variety of different environments and objects including some images that might cause some difficulty such as tennis courts and other sporting activities.

Due to the hardware available to me it was necessary to scale the images down to 128,72p and convert them to grayscale as seen in the make_data_set.py module, this increased the speed of training and reduced the memory requirements, allowing me to load models on an 8GB GPU.

## Building the Neural Network

Again taking into account the hardware available to me and my experience of building similar models in the past I decided to build a model with 3 convolutional layers, and 2 linear layers (the second linear layer being the output layer) as seen in the nn_model.py module. 

For the convolutional layers I started with fairly standard 5x5 Kernels and 2x2 max pooling with the intention to calibrate these parameters in the future if required. As I was working with a convolutional neural network I decided to use the ReLU activation function and avoid the vanishing gradient problem associated with the Sigmoid and Tanh activation functions.

## Training

The train_and_test function I created in the train_and_test.py module allowed me to iterate through each epoch while logging the results. The pass_batch function allowed me to pass batches of the data set to the models reducing memory requirements, and the forward_pass function allowed me to utilise a single function for both training and out of sample testing. 

I used 80% of the data set to train models while using the remaining 20% for out of sample testing. This helped ensure that models generalised rather than over fitting to the training set. I created the test function to perform an out of sample test after each epoch and used the lr_scheduler.ReduceLROnPlateau() function to monitor validation loss which allowed me to reduce the learning rate if the validation loss stagnated. 

As I trained and tested different models It became clear the most effective starting parameters were: a learning rate of 0.000001, a paitence of 2 epochs before reduction, a threshold of 0.01 to only focus on significant changes, and a factor of lr * 0.1 reduction when the threshold of stagnation was met.
