def get_default_arg_parser():
    import argparse
    parser = argparse.ArgumentParser()

    ## actions
    parser.add_argument(
        '--submit', '-s', action='store_true',
        help='make submission')

    parser.add_argument(
        '--cross-validation', '-c', action='store_true',
        help='do cross validation')

    parser.add_argument(
        '--ensemble', '-e', action='store',
        help='do ensemble. set ensemble targets, comma separated')

    parser.add_argument(
        '--call', action='store',
        help='call a function and exit')

    ## values
    parser.add_argument(
        '--name', action='store',
        help='set name for human(if it was omitted, it is generated)')

    parser.add_argument(
        '--xs', action='store', default='xs',
        help='specify train x data')

    parser.add_argument(
        '--xs-sub', action='store', default='xs_sub',
        help='specify test x data')

    parser.add_argument(
        '--model', action='store',
        help='specify model name to create')

    parser.add_argument(
        '--param', dest='param', action='store',
        help='parameters for the model')

    parser.add_argument(
        '--tiny', dest='tiny', action='store_true',
        help='set flag to use tiny data for rapid debugging')

    parser.add_argument(
        '--converter', action='store',
        help='convert xs or xs_sub with given function')

    # misc
    parser.add_argument(
        '--after', dest='after', action='store',
        help='wait until another process stops')

    parser.add_argument(
        '--n-jobs', dest='n_jobs', action='store',
        default="1", help='run parallel jobs if possible')

    return parser
