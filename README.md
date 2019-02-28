# Adapting Auxiliary Losses Using Gradient Similarity
Implementation of ["Adapting Auxiliary Losses Using Gradient Similarity"](https://arxiv.org/abs/1812.02224).

## About
In this repository you will find my trial to recreate some of concepts and (imagenet) experiments presented in mentioned article. Proposed training method seems to improve test accurancy.

## Requirements 
This implementation requires Python 3.x (Conda Python 3.6 environment is recommended).

## Getting started
* (optional, recommended) Create conda virtual enviorment with: `conda create -n env Python=3.6` (swap `-n` with `-p` to create enviorment in project's directory)
* (optional) Activate virtual enviorment `source activate env`
* Install python dependencies `pip install -r requirements.txt` (or use conda package repository).

## About implementation
I have used mostly tensorflow framework to implement proposed training method. Eager execution helps a lot to make it even easier, thanks to its dynamic execution without static graph. I have also used tf.keras.layers API for lavers and model object and keras-contrib Resnet-v2-18 implementation to not risk any errors in architecture itselt.

Due to errors ([Issue #1](https://github.com/keras-team/keras/issues/11927), [Issue #2](https://github.com/tensorflow/tensorflow/issues/21726)) in Batch Normaliation layer, when more than model shares same layers (and their weights) I had to create two separate models for Resnet/Imagenet experiment and then synchronize parameters between models, which is pretty easy, when tf.eager is used.

## Worth to mention
* learning rate could propably be even smaller than 1e-6
* full training should be executed for even more than 20k steps (authors show test accurancy on 200k training steps) - unfortunately I didn't had much time and due to IO/disk performance limitations on google colab I had to execute following code on my XPS 15 9570 (i7-8750h, 16gb ram, gtx1050ti 4gb)
* authors don't mention if their measure average of cosine gradient similarity or in some other way, but I've decided to plot avg cos sim for each batch (1k steps). Results for this plot aren't the same as in paper, so probably they plotted this metric in other way.
* gradient are using only 1 sample at the time, so training can take pretty long time (1/2 day for whole second notebook on my laptop)

## Contents
* [notebooks](https://github.com/szkocot/Adapting-Auxiliary-Losses-Using-Gradient-Similarity/tree/master/notebooks) contains jupyter notebooks with concept and experiments implementation.
* [scripts](https://github.com/szkocot/Adapting-Auxiliary-Losses-Using-Gradient-Similarity/tree/master/scripts) contains implementation of experiments using regular python scripts.
* [bckp](https://github.com/szkocot/Adapting-Auxiliary-Losses-Using-Gradient-Similarity/tree/master/bckp) backup of average test accuracy and gradient cosine similarities.
