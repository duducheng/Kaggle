from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from Base import SearcherBase


class CVSearcher(SearcherBase):
    '''
    Cross validation searcher is not specific for time series
    '''

    def __init__(self, sklearn_model_class, params, scoring=None, method=None,
                 n_randomized_search=200, cv=5):
        super(CVSearcher, self).__init__(sklearn_model_class, params, method=method,
                                         n_randomized_search=n_randomized_search,
                                         cv=cv, scoring=scoring)

    def fit(self, X, Y):
        if self.method == 'Grid':
            self.__searcher = GridSearchCV(estimator=self.ml_class(), param_grid=self.search_space,
                                           scoring=self.scoring, cv=self.cv, refit=True)
        elif self.method == 'Randomized' or self.method is None:
            self.__searcher = RandomizedSearchCV(estimator=self.ml_class(), param_distributions=self.search_space,
                                                 scoring=self.scoring,
                                                 n_iter=self.n_randomized_search, cv=self.cv, refit=True)
        else:
            raise ValueError('CVSearcher only support GridSearch and RandomizedSearch')
        self.__searcher.fit(X, Y)
        print("Best: %s" % (self.__searcher.best_estimator_))
        return self

    def predict(self, X):
        return self.__searcher.predict(X)

    def get_scores(self):
        return self.__searcher.grid_scores_
