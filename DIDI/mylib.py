import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import make_scorer
from dateutil.parser import parse as dateutil_parse

DISTRICTS = {'4725c39a5e5f4c188d382da3910b3f3f', '82cc4851f9e4faa4e54309f8bb73fd7c', 'fff4e8465d1e12621bc361276b6217cf',
             '4b7f6f4e2bf237b6cc58f57142bea5c0', 'fc34648599753c9e74ab238e9a4a07ad', 'd4ec2125aff74eded207d2d915ef682f',
             'a5609739c6b5c2719a3752327c5e33a7', '74c1c25f4b283fa74a5514307b0d0278', '87285a66236346350541b8815c5fae94',
             'b05379ac3f9b7d99370d443cfd5dcc28', '91690261186ae5bee8f83808ea1e4a01', '929ec6c160e6f52c20a4217c7978f681',
             '364bf755f9b270f0f9141d1a61de43ee', '90c5a34f06ac86aee0fd70e2adce7d8a', '1cbfbdd079ef93e74405c53fcfff8567',
             '8bb37d24db1ad665e706c2655d9c4c72', 'd05052b4bda7662a084f235e880f50fa', '2407d482f0ffa22a947068f2551fe62c',
             'dd8d3b9665536d6e05b29c2648c0e69a', '62afaf3288e236b389af9cfdc5206415', 'ca064c2682ca48c6a21de012e87c0df5',
             '2920ece99323b4c111d6f9affc7ea034', '1afd7afbc81ecc1b13886a569d869e8a', '52d7b69796362a8ed1691a6cc02ddde4',
             'b26a240205c852804ff8758628c0a86a', '38d5ad2d22b61109fd8e7b43cd0e8901', 'f47f35242ed40655814bc086d7514046',
             '1ecbb52d73c522f184a6fc53128b1ea1', '52e56004d92b8c74d53e1e42699cba6f', '2350be163432e42270d2670cb3c02f80',
             'b702e920dcd2765e624dc1ce3a770512', 'a814069db8d32f0fa6e188f41059c6e1', '74ec84f1cf75cf89ae176c8c6ceec5ba',
             'ba32abfc048219e933bee869741da911', 'f2c8c4bb99e6377d21de71275afd6cd2', '52a4e8aaa12f70020e889aed8fd5ddbc',
             'bf44d327f0232325c6d5280926d7b37d', '3a43dcdff3c0b66b1acb1644ff055f9d', '2301bc920194c95cf0c7486e5675243c',
             'c9f855e3e13480aad0af64b418e810c3', '08f5b445ec6b29deba62e6fd8b0325a6', '49ac89aa860c27e26c0836cb8dab2df2',
             '73ff8ef735e1d68f0cdcbb84d788f2b6', 'c4ec24e0a58ebedaa1661e5c09e47bb5', '4f8d81b5c31af5d1ba579a65ddc8a5cb',
             '3e12208dd0be281c92a6ab57d9a6fb32', '44c097b7bd219d104050abbafe51bd49', '307afa4120c590b3a46cf4ff5415608a',
             '4b9e4cf2fbdc8281b8a1f9f12b80ce4d', 'f9280c5dab6910ed44e518248048b9fe', 'cb6041cc08444746caf6039d8b9e43cb',
             '693a21b16653871bbd455403da5412b4', 'd524868ce69cb9db10fc5af177fb9423', '4f4041f7db0c7f69892d9b74c1a7efa1',
             '445ff793ebd3477d4a2e0b36b2db9271', '825a21aa308dea206adb49c4b77c7805', '58c7a4888306d8ff3a641d1c0feccbe3',
             '825c426141df01d38c1b9e9c5330bdac', 'de092beab9305613aca8f79d7d7224e7', 'a735449c5c09df639c35a7d61fad3ee5',
             '08232402614a9b48895cc3d0aeb0e9f2', '1c60154546102e6525f68cb4f31e0657', '7f84bdfc2b6d4541e1f6c0a3349e0251',
             'd5cb17978de290c56e84c9cf97e63186', '8316146a6f78cc6d9f113f0390859417', '0a5fef95db34383403d11cb6af937309'}

NON_TRAFFIC_DISTRICTS = {'c4ec24e0a58ebedaa1661e5c09e47bb5'}

WEATHER = {1, 2, 3, 4, 6, 8, 9}


class myStandardScaler(skStandardScaler):
    '''
    Only use fit_transform and tranform, specific for dealing with pd.DataFrame.
    Only scale the numerical features.
    '''

    def fit_transform(self, X):
        Xnumerical = X[X.columns[X.dtypes != bool]]
        Xdummy = X[X.columns[X.dtypes == bool]]
        scaledXnumerical = super(myStandardScaler, self).fit_transform(Xnumerical)
        Xnumerical = pd.DataFrame(scaledXnumerical, index=Xnumerical.index, columns=Xnumerical.columns)
        return pd.concat([Xnumerical, Xdummy], axis=1)

    def transform(self, X):
        Xnumerical = X[X.columns[X.dtypes != bool]]
        Xdummy = X[X.columns[X.dtypes == bool]]
        scaledXnumerical = super(myStandardScaler, self).transform(Xnumerical)
        Xnumerical = pd.DataFrame(scaledXnumerical, index=Xnumerical.index, columns=Xnumerical.columns)
        return pd.concat([Xnumerical, Xdummy], axis=1)


def process_order(order):
    def pclass(p):
        class_set = range(5, 20, 5) + range(20, 100, 10) + range(100, 501, 100)
        idx_set = [p > cls for cls in class_set]
        return idx_set.index(False) if sum(idx_set) != len(idx_set) else len(idx_set)

    order['timeslot'] = order['Time'].map(lambda x: (x.hour * 60 + x.minute) / 10 + 1)
    order['datetimeslot'] = order['Time'].map(lambda x: x.year * 10000 + x.month * 100 + x.day) * 1000 + \
                            order['timeslot']
    order['pclass'] = order['Price'].map(lambda x: pclass(x))
    order = pd.concat([order, pd.get_dummies(order['pclass'], 'pclass').applymap(lambda x: {1.0: True, 0.0: False}[x])],
                      axis=1)
    return order


def process_traffic(traffic):
    traffic['timeslot'] = traffic['datetime'].map(lambda x: (x.hour * 60 + x.minute) / 10 + 1)
    traffic['datetimeslot'] = traffic['datetime'].map(lambda x: x.year * 10000 + x.month * 100 + x.day) * 1000 + \
                              traffic['timeslot']
    return traffic


def process_weather(weather):
    weather['timeslot'] = weather.index.map(lambda x: (x.hour * 60 + x.minute) / 10 + 1)
    weather['datetimeslot'] = pd.Series(
        weather.index.map(lambda x: x.year * 10000 + x.month * 100 + x.day), dtype=np.int64,
        index=weather.index) * 1000 + weather['timeslot']
    target_weather = ['weather_' + str(i) for i in WEATHER]
    weather_dummy = pd.get_dummies(weather['weather'], 'weather').loc[:, target_weather].fillna(0)
    weather_dummy = weather_dummy.applymap(lambda x: {1.0: True, 0.0: False}[x])
    return pd.concat([weather_dummy, weather.drop('weather', axis=1)], axis=1)


def map_order_group(group):
    res = pd.Series()
    res['request'] = group['request'].count()
    res['answer'] = group['request'].sum()
    res['price_avg'] = group['Price'].mean()
    pclass_cols = filter(lambda x: x[:7] == 'pclass_', group.columns)
    pclass_values = group[pclass_cols].sum()
    res = pd.concat([res, pclass_values])
    return res


def get_order_group(order):
    order_group = dict()
    for district in DISTRICTS:
        tmp = order[order['start_district_hash'] == district]
        order_group[district] = tmp.groupby('datetimeslot').apply(lambda g: map_order_group(g))
    return order_group


def get_traffic_group(traffic):
    traffic_group = dict()
    for district in (DISTRICTS - NON_TRAFFIC_DISTRICTS):
        tmp = traffic.ix[district]
        traffic_group[district] = tmp.groupby('datetimeslot').apply(
            lambda g: g[['level_%s' % (level + 1) for level in range(4)]].sum())
    return traffic_group


def XY_order(order_group, dts):
    '''
    This funciton will not consider the first 3 time slot, for all the days.
    It could be changed when test data set changes.
    '''
    timeslot = pd.get_dummies(dts.map(lambda x: int(x % 1000)), 'timeslot')
    # target_timeslot = ['timeslot_'+str(i+1) for i in range(144)]
    target_timeslot = ['timeslot_' + str(i + 1) for i in range(144)]
    timeslot = pd.DataFrame(timeslot, columns=target_timeslot).fillna(0)
    timeslot.index = dts
    timeslot = timeslot.applymap(lambda x: {1.0: True, 0.0: False}[x])
    train = pd.concat([order_group.ix[dts - 1].rename(columns=lambda x: '1_' + x, index=lambda x: x + 1),
                       order_group.ix[dts - 2].rename(columns=lambda x: '2_' + x, index=lambda x: x + 2),
                       order_group.ix[dts - 3].rename(columns=lambda x: '3_' + x, index=lambda x: x + 3),
                       timeslot], axis=1)
    train = train.fillna(0)
    test = (order_group['request'].ix[train.index] - order_group['answer'].ix[train.index])
    return [train, test]


def XY_order_traffic(district, order_group, traffic_group, dts):
    '''
    This funciton will not consider the first 3 time slot (or more), for all the days.
    It could be changed when test data set changes.
    '''
    timeslot = pd.get_dummies(dts.map(lambda x: int(x % 1000)), 'timeslot')
    target_timeslot = ['timeslot_' + str(i + 1) for i in range(144)]
    timeslot = pd.DataFrame(timeslot, columns=target_timeslot).fillna(0)
    timeslot.index = dts
    timeslot = timeslot.applymap(lambda x: {1.0: True, 0.0: False}[x])
    traffic_group_lst = [traffic_group[district].ix[dts - 1].rename(columns=lambda x: '1_' + x, index=lambda x: x + 1),
                         traffic_group[district].ix[dts - 2].rename(columns=lambda x: '2_' + x, index=lambda x: x + 2),
                         traffic_group[district].ix[dts - 3].rename(columns=lambda x: '3_' + x, index=lambda x: x + 3)] \
        if district in traffic_group else []
    order_group_lst = [order_group[district].ix[dts - 1].rename(columns=lambda x: '1_' + x, index=lambda x: x + 1),
                       order_group[district].ix[dts - 2].rename(columns=lambda x: '2_' + x, index=lambda x: x + 2),
                       order_group[district].ix[dts - 3].rename(columns=lambda x: '3_' + x, index=lambda x: x + 3)]
    train = pd.concat(traffic_group_lst + order_group_lst + [timeslot], axis=1)
    train = train.fillna(0)
    test = (order_group[district]['request'] - order_group[district]['answer']).ix[train.index].fillna(0)
    return [train, test]


def mymetrics(target, predict):
    return ((target - predict).abs() / target).replace(np.inf, 0).mean()


def prediction2submit(prediction, cluster_map):
    res = []
    for district in prediction:
        res.append(pd.DataFrame({'prediction': prediction[district], 'district': district}))
    res_1 = pd.concat(res)
    #     res_1['district'] = res_1['district'].map(lambda x: cluster_map.ix[x,0])
    res_2 = pd.DataFrame()
    res_2['district'] = res_1['district'].map(lambda x: cluster_map.loc[x, 'district_id'])
    res_2['dts'] = res_1.index.map(lambda x: '{0}-{1}'.format(dateutil_parse(str(x / 1000)).date(), x % 1000))
    res_2['prediction'] = res_1['prediction']
    res_2['dts_sort'] = res_2.index
    return res_2.sort_values(['dts_sort', 'district']).drop('dts_sort', axis=1)


class Search_Model(object):
    def __init__(self, model_base_class):
        self.__model_base_class = model_base_class

    def fit(self, search_params, X, Y):
        self.__searcher = GridSearchCV(estimator=self.__model_base_class(),
                                       param_grid=search_params,
                                       scoring=make_scorer(mymetrics, greater_is_better=False),
                                       cv=5, n_jobs=2)
        self.__searcher.fit(X, Y)
        self.best_estimator = self.__searcher.best_estimator_.fit(X, Y)
        prediction = pd.Series(self.best_estimator.predict(X), index=Y.index)
        print "Best Params:", self.__searcher.best_params_
        print 'CV score:', -self.__searcher.best_score_
        print "Fit (R2) score:", self.best_estimator.score(X, Y)
        print "The metrics:", mymetrics(Y, prediction)
        return self

    def predict(self, X):
        return self.best_estimator.predict(X)
