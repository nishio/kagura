"""
safetynet

- When unexpected error occurs, enter pdb.
  It offers a chance to save expensive models before process dies.

- Listen SIGUSR1 and when it comes enter pdb.
  When you have a long-running process and you want to watch variables or change values of them,
  you can enter pdb by 'kill -USR1 <pid>' and run codes interactively. 

USAGE:
from kagura.safetynet import safetynet
if __name__ == "__main__":
    safetynet(main)

This script is also a runnable sample.
"""

import traceback
import sys
import pdb
import logging
import signal

def safetynet(func):
    listen_signal()
    try:
        func()
    except:
        type, value, tb = sys.exc_info()
        logging.debug('', exc_info=sys.exc_info())
        traceback.print_exc()
        pdb.post_mortem(tb)


def enter_pdb(signum, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


def listen_signal():
    signal.signal(signal.SIGUSR1, enter_pdb)


def main():
    import time
    i = 1
    while i:
        i += 1
        time.sleep(1)
        # send SIGUSR1 to watch *i*, try 'i = -1' and continue
    print 'finished'


if __name__ == "__main__":
    safetynet(main)

