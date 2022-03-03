# demo_vqa
A Backup for VQA System


This is the code for Visual Question Answering (VQA) System.

The code has two parts. One is a simple demo for presentation. And another is the back end for VQA System.

-------------------

## Demo for Presentation
__Python(Flask), HTML__

It is an easy part. I use very basic features of Flask to have an application for presentation.

Basically I have a server.py to connect the back-end and front-end. The server is the starter class(or start point / entry) for the program.

Additionally, I have a HTML page (static/demo2.html) to show everything when I was on the presentation.

--------------------

## VQA System
__Python(Pytorch)__

It is the big part of this program. 

As part of an VQA question(https://visualqa.org/), I built a VQA System with CLEVR dataset(https://cs.stanford.edu/people/jcjohns/clevr/). 

The algorithm and deep learning structure are from [Compositional Attention Networks for Machine Reasoning (ICLR 2018)](https://arxiv.org/pdf/1803.03067.pdf)

It is a 512 hidden layers and 12 iteration memory "cells" deep learning model. With ResNet preprocessing the images and GloVe + LSTM preprocessing the natural languages. 

The best accuracy based on test data set is 96.95%.

Part of this VQA system is built according to VQA System built from Media Intelligence Lab, Hangzhou Dianzi University.


------------------------
