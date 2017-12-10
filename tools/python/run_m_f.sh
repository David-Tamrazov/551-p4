#!/bin/bash

echo "Single-task learning FASHION"
python2 cnn_single.py --is_fashion --learning_rate=.001 --fresh --epoch=25
python2 cnn_single.py --is_fashion --learning_rate=.00001 --epoch=25

echo "Single-task learning MNIST"
python2 cnn_single.py --learning_rate=0.001 --fresh --epoch=25
python2 cnn_single.py --learning_rate=0.00001 --epoch=25

