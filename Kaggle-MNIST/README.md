# [Kaggle MNIST](https://www.kaggle.com/c/digit-recognizer)

This "easy project" is to test my convnet architecture, firstly used on [NotMNIST](http://yaroslavvb.blogspot.fr/2011/09/notmnist-dataset.html) dataset, which is a dataset more messy than MNIST, where I get a test accuracy of 94.2% @step=9999 (and it seems to still have better result as step increases).

You can also find a working notebook "CNN on MNIST with Tensorflow" on [Kaggle Script](https://www.kaggle.com/jianchengyang/digit-recognizer/cnn-on-mnist-with-tensorflow), where you can directly fork the notebook and run remotely.

***Thanks to [TensorFlow](http://www.TensorFlow.org), even I can build very powerful convnet within very brief code. : )***

# The architecture of myConvnet.
"myConvnet" is very closed to LeNet.
### -> conv(5,5,16) stride=1 padding='SAME' dropout=0.8 -> max pooling(2x2,2) -> relu6
### -> conv(5,5,32) stride=1 padding='SAME' dropout=0.8 -> max pooling(2x2,2) -> relu6
### -> full connected relu6 (192) dropout=0.5
### -> full connected relu6 (64) dropout=0.5
### -> softmax

When training, I used dropout for the purpose of regularization (stop dropout when testing).

# Step accuracy:
1. 1000: 0.96371, Validation: 96.2%
2. 3000: 0.97357, Validation: 97.5%
3. 7000: 0.98471, Validation: 98.4%
4. 9000: 0.98486, Validation: 98.7%
5. 10000: 0.98543, Validation: 98.6%
6. 21000: 0.98857, Validation: 99.3%
7. 28000: 0.99143, Validation: 99.5%
8. 36000: 0.99057, Validation: 99.6%, which seems some overfitting even with dropout.
