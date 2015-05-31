"""
Wait until the specified process is terminated.
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


def listen(args):
    file(QUEUE_NAME, 'a').write('%s\n' % os.getpid())
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


