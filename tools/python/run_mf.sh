!/bin/bash

echo "Mutli-task learning"
python2 cnn_multi.py --epoch 25 --test m --train mf --learning_rate 0.001
python2 cnn_multi.py --epoch 25 --test m --train mf --learning_rate 0.00001 --load_model

echo "Single-task learning FASHION"
python2 cnn_single.py --is_fashion --learning_rate=.001 --load_multi --epoch=25
python2 cnn_single.py --is_fashion --learning_rate=.00001 --epoch=25

echo "Single-task learning MNIST"
python2 cnn_single.py --learning_rate=0.001 --load_multi --epoch=25
python2 cnn_single.py --learning_rate=0.00001 --epoch=25
