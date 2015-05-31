#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This script is a template for a workbench script.
"""
from kagura.getlogger import logging
from kagura.safetynet import safetynet
from kagura.utils import start_end_log
from kagura.getarg import get_args
from kagura import processqueue

@start_end_log
def main():
    args = get_args()
    processqueue.listen(args)

    if args.call_function:
        f = globals()[args.call_function]
        f()

    if args.cross_validation:
        do_cross_validation()

    if args.submit:
        make_submission()

    if args.ensemble:
        ensemble()


if __name__ == "__main__":
    safetynet(main)
