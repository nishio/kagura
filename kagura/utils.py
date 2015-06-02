"""
utilities
"""

def second2human(x):
    "given seconds return human-readable string"
    s_hour = x // 3600
    s_min = x // 60 % 60
    s_sec = int(x) % 60
    ret = ""
    if s_hour:
        ret += "%dh " % s_hour
    if s_hour or s_min:
        ret += "%dm " % s_min
    ret += "%ds" % s_sec
    return ret


import sys
import time
import logging
class HumaneElapse(object):
    def __init__(self, msg='start'):
        if msg:
            logging.info(msg)
        self.start_time = time.time()

    def end(self):
        self.elapse = time.time() - self.start_time
        logging.info("elapse %s", second2human(self.elapse))

    def lap(self):
        self.elapse = time.time() - self.start_time
        return self.elapse

    def get_human(self):
        return second2human(self.elapse)

def stratified_split(xs, ys, nfold=10):
    """
    USAGE:
    train_xs, test_xs, train_ys, test_ys = stratified_split(xs, ys)
    """
    from sklearn.cross_validation import StratifiedKFold
    train, test = StratifiedKFold(ys, nfold).__iter__().next()
    return xs[train], xs[test], ys[train], ys[test]
