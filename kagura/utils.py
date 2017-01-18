"""
utilities
=========

"""

def stratified_split(xs, ys, nfold=10):
    """
    USAGE:
    train_xs, test_xs, train_ys, test_ys = stratified_split(xs, ys)
    """
    from sklearn.cross_validation import StratifiedKFold
    train, test = StratifiedKFold(ys, nfold).__iter__().next()
    return xs[train], xs[test], ys[train], ys[test]


def one_hot_ize(df, col, prefix=None, keep_original=False):
    "take DataFrame and convert specified column to one-hot representation"
    import pandas as pd
    one_hot = pd.get_dummies(df[col], prefix=prefix)
    if not keep_original:
        df = df.drop(col, axis=1)
    df = df.join(one_hot)
    return df


from collections import defaultdict
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
def quick_cv(X, Y, seed=1234):
    "do quick cross validation"
    np.random.seed(seed)
    score = defaultdict(list)
    for train, test in StratifiedShuffleSplit(Y):
        trainX = X[train]
        trainY = Y[train]
        testX = X[test]
        testY = Y[test]

        m = LogisticRegression()
        m.fit(trainX, trainY)
        score['LR'].append(m.score(testX, testY))

        m = KNeighborsClassifier()
        m.fit(trainX, trainY)
        score['KNN'].append(m.score(testX, testY))

        m = DecisionTreeClassifier()
        m.fit(trainX, trainY)
        score['DT'].append(m.score(testX, testY))

    def show(name):
        s = score[name]
        return "{} {:.2f}(+-{:.2f})".format(name, np.mean(s), np.std(s) * 2)

    print ", ".join(show(name) for name in sorted(score))


import time
class Digest(object):
    "print a line per a second to avoid printing overhead"
    def __init__(self, elapse=1):
        self.starttime = time.time()
        self.lasttime = time.time()
        self.num_digested = 0
        self.elapse = elapse

    def digest(self, msg):
        t = time.time()
        if t - self.lasttime < self.elapse:
            self.num_digested += 1
            return 0

        print "{} ({} message digested)".format(
            msg, self.num_digested)
        ret = self.num_digested
        self.num_digested = 0
        self.lasttime = t
        return ret

def to_bytes(array):
    def crop(v):
        v = int(v)
        if v < 0: v = 0
        if v > 255: v = 255
        return chr(v)
    return "".join(map(crop, array))


def from_corner(size):
    for xy in range(size * size):
        for x in range(size):
            y = xy - x
            if y >= size: continue
            if y < 0: break
            yield (x, y)
