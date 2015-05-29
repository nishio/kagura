"""
not mature
"""

def wait(pid):
    """ Check For the existence of a unix pid. """
    import os, time
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        else:
            time.sleep(60)

def args_after():
    if args.after:
        if args.after == "auto":
            pid = int(subprocess.check_output(
                "pgrep python", shell=True).split()[-2])
        else:
            pid = int(args.after)
        logging.info('waiting %s', pid)
        wait(pid)
        logging.info(
            "finish waiting. continue command: %s",
            " ".join(sys.argv))

