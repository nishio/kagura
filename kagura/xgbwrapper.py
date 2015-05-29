"""
XGBoost wrapper to fit Scikit Learn interfaces
"""

import xgboost as xgb
class XGBWrapper(object):
    def __init__(self):
        pass

    def fit(self, xs, ys):
        # ys = 1..(C-1)
        num_class = max(ys)
        ys = ys - 1
        #dtrain = xgb.DMatrix(xs[:100], label=ys[:100])
        dtrain = xgb.DMatrix(xs, label=ys)

        # setup parameters for xgboost
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softprob'
        # scale weight of positive examples
        param['eta'] = 0.1  # 0.01
        param['max_depth'] = 9
        param['sub_sample'] = 0.9
        param['silent'] = 1
        param['nthread'] = 1
        param['num_class'] = num_class
        #param['eval_metric'] = 'auc'
        param['eval_metric'] = 'merror'
        self.param = param
        watchlist = [(dtrain, 'train')]
        num_boost_round = 400
        if 1:
            bst = xgb.train(
                param, dtrain, num_boost_round=num_boost_round,
                evals=watchlist,
            )
        else:
            bst = xgb.Booster(param, [dtrain]) # 2nd arg is cache
            for i in range(num_boost_round):
                # 2nd arg is custom objective function (obj)
                bst.update(dtrain, i, None)
                # 3rd arg is custom evaluation function (feval)
                bst.eval_set(watchlist, i, None)

        #bst.save_model('xgb.model')
        self.bst = bst
        return self

    def predict_proba(self, xs):
        print 'predict proba'
        N = xs.shape[0]
        print N
        xs = xgb.DMatrix(xs)
        print 'ok'
        ypred = self.bst.predict(xs)
        print 'ok2'
        ypred = ypred.reshape(N, self.param['num_class'])
        print 'ok3'
        return ypred

