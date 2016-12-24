"""
find best RF param by hyperopt
"""
import numpy as np
from hyperopt import fmin, tpe, hp
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC

best = 0
def objective(args):
    print args
    global best
    score = []
    for train, test in StratifiedShuffleSplit(_Y):
        trainX = _X[train]
        trainY = _Y[train]
        testX = _X[test]
        testY = _Y[test]

        m = SVC(
            kernel=args[0],
            C=(2 ** -5) * (2 ** args[1]),
            gamma=(2 ** -15) * (2 ** args[2]),
        )
        try:
            m = m.fit(trainX, trainY)
            score.append(m.score(testX, testY))
        except Exception, e:
            print e
            score.append(0)
    s = np.mean(score)
    if s > best:
        print "*", s, args
        best = s
    return -s


def find(X, Y):
    global _X, _Y
    _X = X
    _Y = Y
    space = (
        hp.choice('c0', ['rbf', 'poly', 'sigmoid']),  # 'linear', 
        hp.randint('c1', 20),
        hp.randint('c2', 20),
    )
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=1000)

