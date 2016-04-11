import pandas as pd

class Wrangler(object):

    def __init__(self, raw_dtrain, raw_dtest):
	
        TITLE_AGE = {'Capt': 70.0,
                     'Col': 54.0,
                     'Don': 40.0,
                     'Dona': 39.0,
                     'Dr': 43.571428571428569,
                     'Jonkheer': 38.0,
                     'Lady': 48.0,
                     'Major': 48.5,
                     'Master': 5.4826415094339627,
                     'Miss': 21.774238095238097,
                     'Mlle': 24.0,
                     'Mme': 24.0,
                     'Mr': 32.252151462994838,
                     'Mrs': 36.994117647058822,
                     'Ms': 28.0,
                     'Rev': 41.25,
                     'Sir': 49.0,
                     'the Countess': 33.0}

        TITLE = {'Major': 'Army',
                 'the Countess': 'Upper',
                 'Don': 'Mr',
                 'Sir': 'Upper',
                 'Mlle': 'Upper',
                 'Capt': 'Upper',
                 'Ms': 'Miss',
                 'Jonkheer': 'Upper',
                 'Col': 'Army',
                 'Lady': 'Upper',
                 'Mme': 'Upper',
                 'Dona': 'Upper'}

        self.raw_dtrain = raw_dtrain
        self.raw_dtest = raw_dtest
        self.raw = pd.concat([raw_dtrain, raw_dtest])
        self.processed = pd.DataFrame()
        self.processed[['SibSp', 'Parch', 'Pclass', 'Fare']
                       ] = self.raw[['SibSp', 'Parch', 'Pclass', 'Fare']]
        self.processed['Title'] = self.raw['Name'].map(lambda x: x.split(
            ',')[1].split('.')[0][1:])  # extract "Title" from "Name"
        self.processed['Cabin'] = self.raw['Cabin'].map(lambda x: str(x)[0])
        self.processed['Sex'] = self.raw['Sex'].map(
            lambda x: 0 if x == 'male' else 1)  # male: 0 female: 1

        # deal with NaN and 0
        self.processed['Age'] = self.raw['Age'].groupby(self.processed['Title']).apply(
            lambda g: g.fillna(TITLE_AGE[g.name]))  # average age of Title
        self.processed['Embarked'] = self.raw[
            'Embarked'].fillna('S')  # the most frequent item
        self.processed['Fare'] = self.processed['Fare'].groupby(self.processed['Pclass']).apply(
            lambda g: g.fillna(g.mean()))  # the average Pclass fare
        self.processed['Fare'] = self.processed['Fare'].groupby(self.processed['Pclass']).apply(
            lambda g: g.replace(0, g.mean()))  # the average Pclass fare

        # normalization, and we know test data :)
        self.mean = self.processed[['Age', 'SibSp', 'Parch', 'Fare']].mean()
        self.std = self.processed[['Age', 'SibSp', 'Parch', 'Fare']].std()
        self.processed[['Age', 'SibSp', 'Parch', 'Fare']] = (
            self.processed[['Age', 'SibSp', 'Parch', 'Fare']] - self.mean) / self.std

        # then also merge some rare Title into commom ones
        self.processed['Title'] = self.processed['Title'].map(
            lambda x: TITLE[x] if x in TITLE else x)

        # transfer category feature into dummy feature
        category_Embarked = pd.get_dummies(
            self.processed['Embarked'], prefix='Embarked')
        category_Pclass = pd.get_dummies(
            self.processed['Pclass'], prefix='Pclass')
        category_Cabin = pd.get_dummies(
            self.processed['Cabin'], prefix='Cabin')
        category_Title = pd.get_dummies(
            self.processed['Title'], prefix='Title')
        self.processed = pd.concat(
            [self.processed, category_Embarked, category_Pclass, category_Cabin, category_Title], axis=1)
        # drop features we don't need
        self.processed = self.processed.drop(
            ['Embarked', 'Pclass', 'Cabin', 'Title'], axis=1)

        # export X, y
        self.Xtrain = self.processed.ix[self.raw_dtrain.index, :]
        self.Xtest = self.processed.ix[self.raw_dtest.index, :]
        self.ytrain = self.raw_dtrain['Survived']
