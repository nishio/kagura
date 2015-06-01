"""
XGBoost wrapper to fit Scikit Learn interfaces

if 'test' given, do early stopping


eta: 0.01[1], 0.4-1.0[2], 0.1-0.3[3] default:0.3
max_depth 9[1], 4-10[2], default 6, 5-6[3]
subsample [default=1] 0.4-1[2], 0.9-1.0[3]
colsample_bytree [default=1], 0.4-1[2], 0.9-0.1[3]
num_round 3000[1] 5..35?[2]

[1]
https://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/
[2]
https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov
[3]
My experiments for Poker Rule Induction

"""

import xgboost as xgb

class XGBBinary(object):
    "XGB for binary clustering"
    def __init__(self, num_boost_round=400, test=None, **params):
        self.params = params
        self.num_boost_round = num_boost_round
        if test:
            test_xs, test_ys = test
            test = xgb.DMatrix(test_xs, label=test_ys)
        self.test = test

    def fit(self, xs, ys):
        dtrain = xgb.DMatrix(xs, label=ys)

        # setup parameters for xgboost
        params = self.params
        params['silent'] = 1
        params['nthread'] = 1

        if not params.get('eval_metric'):
            params['eval_metric'] = 'auc'

        if self.test:
            watchlist = [(dtrain, 'train'), (self.test, 'test')]
            early_stopping_rounds = 10
        else:
            watchlist = [(dtrain, 'train')]
            early_stopping_rounds = 0

        bst = xgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.bst = bst

        return self

    def predict_proba(self, xs):
        print 'predict proba'
        N = xs.shape[0]
        xs = xgb.DMatrix(xs)
        pred = self.bst.predict(xs, ntree_limit=self.bst.best_iteration)
        return pred

    def predict(self, xs):
        pred = np.array(self.predict_proba(xs))
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        return pred

     def score(self, xs, ys):
        from sklearn.metrics import log_loss
        pred = self.predict_proba(xs)
        return log_loss(ys, pred)


class XGBWrapper(object):
    "sklearn adapter"
    def __init__(self, num_boost_round=400, test=None, **params):
        self.params = params
        self.num_boost_round = num_boost_round
        if test:
            test_xs, test_ys = test
            test = xgb.DMatrix(test_xs, label=test_ys)
        self.test = test

    def fit(self, xs, ys):
        # When using Softmax, xgboost requires ys must be in [0, num_class)
        num_class = self.params['num_class']
        assert all(0 <= y < num_class for y in ys)

        dtrain = xgb.DMatrix(xs, label=ys)

        # setup parameters for xgboost
        params = self.params

        # use softmax multi-class classification
        params['objective'] = 'multi:softprob'
        params['silent'] = 1
        params['nthread'] = 1

        if not params.get('eval_metric'):
            params['eval_metric'] = 'mlogloss' # logloss for multiclass
            # other candidate: auc, merror, ...

        if self.test:
            watchlist = [(dtrain, 'train'), (self.test, 'test')]
            early_stopping_rounds = 10
        else:
            watchlist = [(dtrain, 'train')]
            early_stopping_rounds = 0

        bst = xgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.bst = bst

        return self

    def predict_proba(self, xs):
        print 'predict proba'
        N = xs.shape[0]
        xs = xgb.DMatrix(xs)
        pred = self.bst.predict(xs, ntree_limit=self.bst.best_iteration)
        pred = pred.reshape(N, self.params['num_class'])
        return pred

    def score(self, xs, ys):
        from sklearn.metrics import log_loss
        pred = self.predict_proba(xs)
        return log_loss(ys, pred)
