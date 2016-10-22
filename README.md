# deep-learning
Deep learning Projects

## 1. Identify nerve structures in ultrasound images of the neck - Kaggle Competition ##

In this project, nurve structures in a dataset of ultrasound images of the neck were correctly identified to effectively insert a patient’s pain management catheter. Convolution Neural nertwork is implemented in Keras. Simple U-Net Architecture is used as a base architecture and it was fine tuned according to the performance on the training data. Training data consists of images where the nerve has been manually annotated by humans.

## 2. Distinguish images of dogs from cats - Kaggle Competition ##

In this project, Dogs vs. Cats classification problem is studied. Convolutional Neural Nets are used to tackle the problem of classification. VGG-16 base architecture has been modified to fit the training dataset of 25K images(12.5K dogs and cats each). Different techniques like Batch normalization, Dropout instead of max pooling, ELU vs ReLU are studied to train the model. Results of trained model from scratch are compared with pre-trained weights of VGG-16 from IMAGENET. Different image augmentations are applied to overcome the problem of overfitting on pre-trained model, since it was trained on 1.2M images from IMAGENET with 1000 classes.
