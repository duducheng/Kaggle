from os import path, listdir
from keras.models import Model, model_from_yaml
from imagenet_utils import preprocess_input
import numpy as np


# # These 2 files was generated for TensorFlow, you should generate them based on your tools.
# # The config file is "~/.keras/keras.json", specify the "image_dim_ordering" and "backend" fields.
# MODEL_STRUCTURE = 'vgg16_notop.yaml'
# MODEL_WEIGHTS = 'vgg16_notop.h5'
files = listdir(path.join(path.dirname(__file__)))
MODEL_STRUCTURE = filter(lambda x: x.endswith('.yaml'), files)
MODEL_WEIGHTS = filter(lambda x: x.endswith('.h5'), files)
assert len(MODEL_STRUCTURE) == 1 and len(
    MODEL_WEIGHTS) == 1, "There should be just one model in the folder"
MODEL_STRUCTURE = MODEL_STRUCTURE[0]
MODEL_WEIGHTS = MODEL_WEIGHTS[0]

locate_path = lambda p: path.abspath(path.join(path.dirname(__file__), p))

_STRUCTURE = locate_path(MODEL_STRUCTURE)
_WEIGHTS = locate_path(MODEL_WEIGHTS)


def load(trainable=False, layer=None):
    '''
    This model can be use as a normal keras model.
    '''
    with open(_STRUCTURE) as f:
        kr = model_from_yaml(f)
    kr.load_weights(_WEIGHTS)
    if layer is not None:
        kr = Model(input=kr.input,
                   output=kr.get_layer(layer).output)
    if trainable:
        for kr_layer in kr.layers:
            kr_layer.trainable = True
    return kr


class FeatureExtracter(object):

    def __init__(self, layer=None):
        model = load(layer=layer)
        self.__model = model

    def extract(self, rgb):
        x = np.expand_dims(rgb, axis=0)
        x = preprocess_input(x)
        return self.__model.predict(x)


def feature_extract(rgb, layer=None):
    '''
    For one time extraction.
    '''
    obj = FeatureExtracter(layer)
    return obj.extract(rgb)
