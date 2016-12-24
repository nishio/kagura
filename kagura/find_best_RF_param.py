"""
find best RF param by hyperopt
"""
import numpy as np
from hyperopt import fmin, tpe, hp
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

best = 0
def objective(args):
    global best
    score = []
    for train, test in StratifiedShuffleSplit(_Y):
        trainX = _X[train]
        trainY = _Y[train]
        testX = _X[test]
        testY = _Y[test]

        m = RandomForestClassifier(
            n_estimators=args[0],
            criterion=args[1],
            max_depth=args[2],
        )
        try:
            m = m.fit(trainX, trainY)
            score.append(m.score(testX, testY))
        except Exception, e:
            print e
            score.append(0)
    s = np.mean(score)
    if s > best:
        print s, args
        best = s
    return -s


def find(X, Y):
    global _X, _Y
    _X = X
    _Y = Y
    space = (
        hp.randint('c0', 100) + 1,
        hp.choice('c1', ['gini', 'entropy']),
        hp.randint('c2', 100) + 1,
    )
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=1000)

