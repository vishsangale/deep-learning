# Deep learning Projects

## 1. Identify nerve structures in ultrasound images of the neck - Kaggle Competition ##

In this project, nurve structures in a dataset of ultrasound images of the neck were correctly identified to effectively insert a patient’s pain management catheter. Convolution Neural nertwork is implemented in Keras. Simple [U-Net Architecture](https://arxiv.org/pdf/1505.04597.pdf) is used as a base architecture and it was fine tuned according to the performance on the training data. Training data consists of images where the nerve has been manually annotated by humans.

## 2. Distinguish images of dogs from cats - Kaggle Competition ##

In this project, Dogs vs. Cats classification problem is studied. Convolutional Neural Nets are used to tackle the problem of classification. [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) as a base architecture has been modified to fit the training dataset of 25K images(12.5K dogs and cats each). Different techniques like Batch normalization, Dropout instead of max pooling, ELU vs ReLU are studied to train the model. Results of trained model from scratch are compared with pre-trained weights of VGG-16 from IMAGENET. Different image augmentations are applied to overcome the problem of overfitting on pre-trained model, since it was trained on 1.2M images from IMAGENET with 1000 classes. Log loss of 0.21584 is achieved(Accuracy 95%).

## 3. Classify handwritten digits using the famous MNIST data ##

The goal of this project is to take an image of a handwritten single digit, and determine what that digit is. Images for this project is taken from the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset. Dataset consists of gray-scale images of hand-drawn digits, from zero through nine of shape 28x28. Simple 4-layer convolutional neural net is implemented  along with max pooling layers to achieve the accuracy of 98.971%.
