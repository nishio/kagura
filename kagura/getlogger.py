"""
get configured logger

USAGE at the beginning of script:

from kagura.getlogger import logging
"""

import logging
import sys
import atexit

logging.basicConfig(
    filename='log.txt',level=logging.DEBUG,
    format='%(asctime)s/(%(process)d) %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("start process: %s", " ".join(sys.argv))

def end():
    logging.info("end process: %s", " ".join(sys.argv))

atexit.register(end)
