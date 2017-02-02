"""
utils on time
"""
import time

def second2human(x, ignore_msec=True):
    """
    given seconds return human-readable string

    >>> second2human(10)
    '10s'
    >>> second2human(100)
    '1m 40s'
    >>> second2human(10000)
    '2h 46m 40s'
    >>> second2human(1.234, ignore_msec=False)
    '1s 234ms'
    """
    s_hour = x // 3600
    s_min = x // 60 % 60
    s_sec = int(x) % 60
    ret = ""
    if s_hour:
        ret += "%dh " % s_hour
    if s_hour or s_min:
        ret += "%dm " % s_min
    ret += "%ds" % s_sec
    if not ignore_msec:
        s_msec = int(x * 1000) % 1000
        ret += " %dms" % s_msec
    return ret


class Stopwatch(object):
    """
    >>> sw = Stopwatch(name="foo", start_now=True, ignore_msec=True)
    start foo
    >>> sw.end()
    end foo: 0s
    """
    def __init__(self, name="stopwatch", start_now=False,
                 ignore_msec=False, to_log=False):
        self.name = name
        self.ignore_msec = ignore_msec
        self.to_log = to_log
        if to_log:
            import kagura.getlogger
            self.logger = kagura.getlogger.get_logger_to_stdout()

        if start_now:
            self.start()

    def _log(self, msg):
        if self.to_log:
            self.logger.info(msg)
        else:
            print(msg)

    def start(self):
        self._log("start {}".format(self.name))
        self.start_time = time.time()

    def end(self):
        self._log("end {}: {}".format(
            self.name, self.get()
        ))

    def get(self):
        elapse = time.time() - self.start_time
        return second2human(elapse, self.ignore_msec)

    def restart(self):
        self.end()
        self.start()

    def __call__(self, f):
        "use as decorator"
        if self.name == "stopwatch":
            self.name = f.__name__
        def _f(*args, **kw):
            self.start()
            ret = f(*args, **kw)
            self.end()
            return ret
        return _f

    def __enter__(self):
        "use with with-statement"
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        "use with with-statement"
        self.end()
        if type:
            raise

@Stopwatch()
def _test_stopwatch():
    pass

def _test_stopwatch2():
    with Stopwatch() as sw:
        sw.restart()

def _test():
    """
    >>> _test_stopwatch()
    start _test_stopwatch
    end _test_stopwatch: 0s 0ms
    >>> _test_stopwatch2()
    start stopwatch
    end stopwatch: 0s 0ms
    start stopwatch
    end stopwatch: 0s 0ms
    """
    import doctest
    doctest.testmod()
    print 'test ok'

if __name__ == '__main__':
    _test()
