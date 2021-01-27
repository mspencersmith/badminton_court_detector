# Badminton Court Detector

In this project I built and trained a convolutional neural network to detect badminton courts in images.

## Technologies
### Python 3.6
* Pytorch
* Matplotlib
* OpenCV
* Numpy

## Building Data Set

In order to ensure the highest level of accuracy from the model it was important to gather a large quantity of images, while at the same time ensuring that the data set was balanced.

I was able to collect 12,000 images of badminton courts by utilising OpenCV's VideoCapture() and read() methods on videos of badminton matches. As a former professional player I had many videos to choose from, along with other videos and stock images. I made sure I captured as many different arenas and angles as possible, and also flipped images to increase the number I had to work with.

For balance I also collected 12,000 random stock images to test and train the model with, ensuring again to gather a variety of different environments and objects including some images that might cause some difficulty such as tennis courts and other sporting activities. 
