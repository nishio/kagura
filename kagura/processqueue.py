"""
Wait until the specified process is terminated.

SAMPLE:
def foo():
    import time
    time.sleep(5)
    logging.info('foo')


def bar():
    logging.info('bar')

COMMAND:
$ python t.py --call=foo --after=auto
$ python t.py --call=bar --after=auto

LOG OUTPUT:
2015-05-31 11:16:30/(17902) start process: t.py --call=foo --after=auto
2015-05-31 11:16:30/(17902) waiting 17832, but it is already finished
2015-05-31 11:16:32/(17916) start process: t.py --call=bar --after=auto
2015-05-31 11:16:32/(17916) waiting 17902
2015-05-31 11:16:36/(17902) foo
2015-05-31 11:16:36/(17902) end process: t.py --call=foo --after=auto
2015-05-31 11:16:36/(17916) finish waiting. continue command: t.py --call=bar --after=auto
2015-05-31 11:16:36/(17916) bar
2015-05-31 11:16:36/(17916) end process: t.py --call=bar --after=auto
"""


import os
import time
import logging
import sys

QUEUE_NAME = 'process_queue.txt'

def is_process_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait(pid):
    """ Check For the existence of a unix pid. """
    while is_process_alive(pid):
        time.sleep(1)


def get_previous_pid():
    if not os.path.exists(QUEUE_NAME):
        file(QUEUE_NAME, 'w')
        return
    pids = file(QUEUE_NAME).readlines()
    if len(pids) < 2:
        return
    return int(pids[-2])


def record_pid():
    if sys.argv[0].endswith('ipython'):
        return
    file(QUEUE_NAME, 'a').write('%s\n' % os.getpid())


def listen(args):
    record_pid()
    if args.after:
        if args.after == "auto":
            pid = get_previous_pid()
        else:
            pid = int(args.after)
        if not pid:
            logging.info('--after specified, but no preceeding process.')
            return
        if not is_process_alive(pid):
            logging.info('waiting %s, but it is already finished', pid)
            return

        logging.info('waiting %s', pid)
        wait(pid)
        logging.info(
            "finish waiting. continue command: %s",
            " ".join(sys.argv))


