# EMNIST-Balanced Handwritten Character Recognition

## Overview

This project aims to recognize handwritten characters using the EMNIST-balanced dataset and LeNet architecture. The EMNIST-balanced dataset consists of 47 classes and a total of 131,600 images. The LeNet architecture is a deep neural network consisting of 7 layers, including 2 convolutional layers and 2 fully connected layers. It is designed specifically for image recognition tasks and has shown great performance on MNIST dataset.

## Data

The EMNIST-balanced dataset contains handwritten characters from A to Z (capital letters), a to z (lowercase letters), and digits from 0 to 9. The dataset is balanced, meaning that each class has an equal number of samples. The dataset is split into a training set with 112,800 images and a test set with 18,800 images. Each image has a resolution of 28 x 28 pixels (https://arxiv.org/pdf/1702.05373.pdf).

## Methods and Ideas

The LeNet architecture is a good choice for this project due to its success on similar datasets, such as the MNIST dataset. It consists of two convolutional layers, two max-pooling layers, and two fully connected layers. The first convolutional layer uses 20 filters, while the second uses 50 filters. Each convolutional layer is followed by a max-pooling layer that reduces the spatial size of the output. Finally, the output is flattened and fed into two fully connected layers.

The model was trained using a batch size of 128 and for 20 epochs. The optimizer used was SGD with learning rate 0.01, and the loss function used was categorical cross-entropy.
The training accuracy and validation accuracy of the final model were as follows:
>
> - Train Accuracy: 0.8949
> - Validation Accuracy: 0.8549 
>

## Usage Instructions

Install the required packages by running the following command:

`pip3 install -r requirements.txt`


Run the following command to evaluate the model on the test set:

`python3 inference.py --input ('/path/to/test_data')`


## Author

This model was trained by [Oksana Bolibok] - oksanabolibok496@gmail.com.
