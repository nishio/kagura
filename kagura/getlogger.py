"""
get configured logger

USAGE at the beginning of script:

from kagura.getlogger import logging
"""

import logging

logging.basicConfig(
    filename='log.txt',level=logging.DEBUG,
    format='%(asctime)s/(%(process)d) %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("start process")
