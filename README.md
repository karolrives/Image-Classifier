# ImageClassifier

Implemented a convolutional neural network to classify images from the CIFAR-10 dataset.

## Installation

There is no necessary libraries to run the code, except the ones included in the Jupyter Notebook. The code runs with no issues using Python 3 or newer versions.

## Project Motivation

This project was part of the Data Science Udacity Program and needed to be completed in order to obtain a certificate.

## File Description

This project is divided in two parts. 
The first part is a Jupyter Notebook, where I implemented an image classifier with PyTorch. The files involved in this part are the following: 

* **Image Classifier Project_Final.ipynb**
* **cat_to_name.json**: Dictionary file. Needed for mapping the name of images.


The second part is a commmand line application for the image classifier. The files involved in this part are the following:

* **train.py**: Trains a new network on a dataset ans save the model as a checkpoint.
Usage:

    `` python train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epoch EPOCH] [--gpu]
                data_directory
 ``

    Run ``python train.py -h`` for more information. 

* **predict.py**: Uses a trained network to predict the class for an input image.
Usage:

    ``python predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  image_path checkpoint
``

    Run ``python predict.py -h`` for more information.

* **utils.py**: Contains functions relating to the creation of the model, loading data, preprocessing images. Used by **train.py** 
and **predict.py**

* **cat_to_name.json**: Dictionary file. Needed for mapping the name of images.

Note: The two parts of this project are independent from each other. 


## Results

The main findings are found in the Jupyter Notebook **Image Classifier Project_Final.ipynb**, where markdown cells were used to walk through all the steps.

