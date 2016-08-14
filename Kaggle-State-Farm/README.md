# [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
I took on project after the deadline of the competition, while due to this project highly related to my other side project ([SODA](http://soda.datashanghai.gov.cn/?lang=en)), I decided to explore the data by myself. Using pretrained [ResNet-50](https://github.com/KaimingHe/deep-residual-networks) provided by [Keras](https://github.com/fchollet/keras/) (personally favorite high level framework for Deep Learning), I got good performance on both validation set and test set.

Since I didn't use stack trick so far (the competition has already ended up, why I need to take so much time on tuning = ,= ), there is still large room to improve, even for the boosting trees model. And considering it's possible to include the driver information in NN models (like embedding), this can also be used to make progress.

# How-To
To run the code, download and unzip the data from Kaggle in "data" folder.

Install Keras (network part) and Graphlab Create (boosting tree part).

The other packages are all included in Anaconda.

# Methodology
|=> Resize the images to 224 x 224 (like the images from ImageNet)

|=> High level features from pretrained CNN (ResNet-50)

|=> Boosting trees to classify.

I tried VGG-16 and ResNet-50 for extracting features, and finally I used ResNet-50, cause it's really really faster when classification. I took the last layer features (i.e. after average pooling), and flatten, which gives 2048 features.

Then I connected the features to a boosting tree model, for 10-class classification.

# Why I didn't use VGG-16
VGG-16 has 3-layer full connected network by the end, so it's not really safe to use the last layer features. I use the features after its last block conv layer, which is 7 x 7 x 512 features. There may be much more possibilities to explore these intermediate featres, while due to my computation limit, I didn't explore too much.

# Time
I used AWS g2.2xlarge to extract the features.

For VGG-16, it took about 0.08s per image. And For ResNet-50,  it tooks about 0.09s per image.

To train on the extracted features, VGG features has 25088 dimensions (sparse), while ResNet has only 2048 features (after average pooling).

I used [Graphlab Create](https://turi.com/) to train boosting tree models locally (on my Surface Book), where on VGG features, it took ~88s to train a 3-depth tree, while on ResNet features it took only ~1.1s per tree. Note Graphlab Create is a commercial software, but you can register for academic use of GraphLab Create for free.

# Tips for feature extraction
You'd better extract the feature once, rather than keep it not trainable in the layers, if you are not trying to keep the feature extraction layer in the middle. The reason is quiet simple, just because for this scale network, it really takes long time for forward prop, you should not let your network do it every time.

Well, with Keras, it's really easy to keep it in your network, but in this case, you should not do that.

# Tools
The package resnet50 is a simple refactor version from Keras [deep-learning-models](https://github.com/fchollet/deep-learning-models/), just remove some data downloads modules. If you want to use it, you should install Keras (1.0.7) from GitHub, not from PyPi.

Graphlab Create uses Xgboost for its boosting tree backend base, so you can also use Xgboost to get same performance. I don't recommend you to use scikit-learn in this case, although I'm really a big fun, it will take really much more time.

# License
MIT
