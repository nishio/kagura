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
def start_end_log(f):
    def _wrap():
        "decorate function to log when start and end"
        logging.info("command: %s", " ".join(sys.argv))
        start_time = time.time()
        logging.info("start: %s", f)
        f()
        logging.info("end: %s", f)
        elapse = time.time() - start_time
        logging.info("elapse %s", second2human(elapse))
        logging.info("end command: %s", " ".join(sys.argv))
    return _wrap


def stratified_split(xs, ys, nfold=10):
    """
    USAGE:
    train_xs, test_xs, train_ys, test_ys = stratified_split(xs, ys)
    """
    from sklearn.cross_validation import StratifiedKFold
    train, test = StratifiedKFold(ys, nfold).__iter__().next()
    return xs[train], xs[test], ys[train], ys[test]
