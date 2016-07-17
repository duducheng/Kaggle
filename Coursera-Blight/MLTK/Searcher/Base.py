from abc import ABCMeta, abstractmethod
import numpy as np

'''
Searcher provide a limited API than sklearn searcher,
while can be useful in the time series project.
'''


class SearcherBase(object):
    '''
    Can add monto carlo searcher, and boootstrap searcher
    '''

    __metaclass__ = ABCMeta

    def __init__(self, ml_class, search_space, **kwargs):
        self.ml_class = ml_class
        self.search_space = search_space
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def fit(self, X, Y):
        '''
        Search best param in the search space.
        '''

    @abstractmethod
    def predict(self, X):
        '''
        With refit.
        '''

    @abstractmethod
    def get_scores(self):
        '''
        Same as sklearn's searcher.grid_scores_: list of named tuples
        * parameters, a dict of parameter settings
        * mean_validation_score, the mean score over the cross-validation folds
        * cv_validation_scores, the list of scores for each fold
        '''

    def report(self, n_top=10):
        grid_scores = self.get_scores()
        top_scores = sorted(grid_scores, key=lambda x: x[
            1], reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("========================= Model with rank: {0} =========================".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                np.mean(score.cv_validation_scores), np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print('^_^')
