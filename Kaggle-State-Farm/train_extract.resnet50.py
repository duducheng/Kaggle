import time
import resnet50
import PIL.Image
import os.path
import numpy as np
import pandas as pd
from config import PATH, TRAINING

IMGS = os.path.join(PATH, "imgs")
TARGET = 'train'
OUTPUT_SHAPE = (224, 224)

train_info = pd.read_csv(os.path.join(PATH, 'driver_imgs_list.csv'))

train_info['path'] = train_info.apply(lambda row: os.path.join(
    IMGS, TARGET, row['classname'], row['img']), axis=1)

extracter = resnet50.FeatureExtracter()

PIL.Image.Image.resized_array = lambda img, shape=None: np.array(
    img if shape is None else img.resize(shape))

target = train_info.path[:5]  # for test
# target = train_info.path
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

train_info['conv'] = target.map(extract)


train_info.to_pickle(TRAINING)

print "==============================="
print "Take", time.time() - now
