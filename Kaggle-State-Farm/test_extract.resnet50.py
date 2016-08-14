import os
import numpy as np
import pandas as pd
import resnet50
import PIL.Image
from config import PATH, TESTING
import time

IMGS = os.path.join(PATH, "imgs")
TARGET = 'test'
OUTPUT_SHAPE = (224, 224)

test_list = pd.DataFrame({'img': os.listdir(os.path.join(IMGS, TARGET))})
test_list['path'] = test_list.img.map(
    lambda p: os.path.join(os.path.join(IMGS, TARGET), p))

extracter = resnet50.FeatureExtracter()
PIL.Image.Image.resized_array = lambda img, shape=None: np.array(
    img if shape is None else img.resize(shape))


target = test_list.path[:5]  # for test
# target = test_list.path
length = len(target)
idx = 1
now = time.time()


def extract(path):
    im = PIL.Image.open(path)
    global idx, length, extracter, now
    ret = extracter.extract(im.resized_array(OUTPUT_SHAPE).astype(np.float64))
    print "Processing {0:d}/{1:d}, totally using {2:.2f}".format(idx, length, time.time() - now)
    idx += 1
    return ret

test_list['conv'] = target.map(extract)

test_list.to_pickle(TESTING)

print "==============================="
print "Take", time.time() - now
