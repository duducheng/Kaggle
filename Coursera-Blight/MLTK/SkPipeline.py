from MLTK.Scaler import Scaler
from MLTK.Searcher import CVSearcher


def SkPipeline(skbase):
    class SearchScale(skbase):
        def __init__(self, scoring=None, method=None, n_randomized_search=100, cv=3, **kwargs):
            self.__scaler = Scaler()
            for k in kwargs:
                if not hasattr(kwargs[k], '__iter__'):
                    kwargs[k] = [kwargs[k]]
            self.__search_params = kwargs
            self.__scoring = scoring
            self.__method = method
            self.__n_randomized_search = n_randomized_search
            self.__cv = cv

        def fit(self, X, y):
            X = self.__scaler.fit_transform(X)
            self.__searcher = CVSearcher(skbase, self.__search_params, self.__scoring,
                                         self.__method, self.__n_randomized_search, self.__cv)
            self.__searcher.fit(X, y)
            return self

        def predict(self, X):
            X = self.__scaler.transform(X)
            return self.__searcher.predict(X)

    return SearchScale


if __name__ == "__main__":
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from TSeries.MLTK.Scaler import Scaler
    from TSeries.MLTK.Searcher import CVSearcher

    from Flow import Flow
    from Wrangler import Wrangler

    tflow = Flow().fetch("test").clean()
    wrangler = Wrangler(tflow,
                        {"spot_ahead": 7, "consumption_to_ahead": 2, "generation_ahead": 1},
                        {"spot_rollings": [15, 30], "consumption_rollings": [7, 15], "generation_rollings": [3, 7, 15]})
    Xtrain, Ytrain, Xtest, Ytest = wrangler.train_test("20120101", "20141231", "20150101", "20160105", "D",
                                                       predict_ahead=1)

    scaler = Scaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    model = SkPipeline(KernelRidge)(**{'kernel': ['poly'], 'alpha': np.random.rand(300) * 7 + 0.5})
    model.fit(Xtrain, Ytrain)
    print model.predict(Xtrain)
