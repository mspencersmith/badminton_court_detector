# Badminton Court Detector

In this project I built and trained a convolutional neural network (Pytorch model) to detect badminton courts in images.

## Technologies
### Python 3.6
* Pytorch
* Matplotlib
* OpenCV
* Numpy

## Gathering Data Set

In order to ensure the highest level of accuracy from the model it was important to gather a large quantity of images, while at the same time ensuring that the data set was balanced.

I was able to collect 12,000 images of badminton courts by utilising OpenCV's VideoCapture() and read() methods on videos of badminton matches. As a former professional player I had many videos to choose from, along with other videos and stock images. I made sure I captured as many different arenas and angles as possible, and also flipped images to increase the number I had to work with.

For balance I also collected 12,000 random stock images to test and train the model with, ensuring again to gather a variety of different environments and objects including some images that might cause some difficulty such as tennis courts and other sporting activities.

Due to the hardware available to me it was necessary to scale the images down to 128,72p and convert them to grayscale as seen in the make_data_set.py module, this increased the speed of training and reduced the memory requirements, allowing me to load the model on an 8GB GPU.
