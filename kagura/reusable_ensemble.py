"""
reusable ensemble
"""
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

def setup(X, Y, model_generators):
    "setup reusable ensemble"
    blendfeed = []
    models_s = []
    def add_model(M):
        bf, ms = reusable_ens_core(X, Y, M)
        blendfeed.append(bf)
        models_s.append(ms)

    for m in model_generators:
        add_model(m)

    blender = LogisticRegression()
    a = np.hstack(blendfeed)
    blender.fit(a, Y)

    def f(Xsub):
        bs = []
        M = len(Xsub)
        K = 2 # len(Y[0])
        for models in models_s:
            Xsubconv = np.zeros((M, K))
            for m in models:
                Xsubconv += m.predict_proba(Xsub)
            bs.append(Xsubconv)
        b = np.hstack(bs)
        return blender.predict_proba(b)

    return f


def reusable_ens_core(X, Y, model):
    models = []
    sss = StratifiedShuffleSplit(Y)
    K = 2 # len(Y[0])
    N = len(X)
    D = len(X[0])
    Xconv = np.zeros((N, K))
    for train, test in sss:
        trainX = X[train]
        trainY = Y[train]
        testX = X[test]
        testY = Y[test]

        m = model()
        m.fit(trainX, trainY)
        models.append(m)
        Xconv[test] = m.predict_proba(testX)
    return Xconv, models

