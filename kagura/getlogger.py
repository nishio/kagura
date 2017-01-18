"""
get configured logger

USAGE at the beginning of script:

(obsolete)from kagura.getlogger import logging

It is obsolete because it is not good practice.


"""

from logging import getLogger, StreamHandler, DEBUG

def get_logger_to_file():
    import logging
    logging.basicConfig(
        filename='log.txt',level=logging.DEBUG,
        format='%(asctime)s/(%(process)d) %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    return getLogger("kagura")
logger = get_logger_to_file()  # it write on file silently


def get_logger_to_stderr():
    logger = getLogger("kagura.stderr")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger  # it writes on file and stderr


def get_logger_to_stdout():
    import sys
    logger = getLogger("kagura.stdout")
    handler = StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger  # it writes on file and stdout


def log_process_life():
    import sys
    import atexit
    import logging
    logging.info("start process: %s", " ".join(sys.argv))

    def end():
        logging.info("end process: %s", " ".join(sys.argv))

    atexit.register(end)


def _test():
    import doctest
    doctest.testmod()
    logger =  get_logger_to_file()
    log_process_life()
    logger.debug('hello')

    logger2 =  get_logger_to_stderr()
    logger2.debug('hello again')

    logger.debug('hello three')


if __name__ == '__main__':
    _test()
