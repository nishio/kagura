
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
def start_end_log(f):
    "decorate function to log when start and end"
    logging.info("command: %s", " ".join(sys.argv))
    start_time = time.time()
    logging.info("start: %s", f)
    f()
    logging.info("end: %s", f)
    elapse = time.time() - start_time
    logging.info("elapse %s", second2human(elapse))
    logging.info("command: %s", " ".join(sys.argv))


