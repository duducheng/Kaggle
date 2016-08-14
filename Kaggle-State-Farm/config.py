# Put your data from Kaggle in the $PATH%
# Recommend use kaggle-cli
# sudo pip install kaggle-cli
# kg config -g -u <username> -p <password>
# kg download -c <competition>

PATH = 'data'
_TRAINING = 'train.resnet50.pkl'
_TESTING = 'test.resnet50.pkl'

RESULT = 'result'
GL_MODEL = 'boosting'


import os

for p in (PATH, RESULT):
    if not os.path.exists(p):
        os.mkdir(p)

TRAINING = os.path.join(PATH, _TRAINING)
TESTING = os.path.join(PATH, _TESTING)
