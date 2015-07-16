""" Auxiliary functions for development test suite.
"""

# standard library
import logging
import socket
import shutil
import glob
import os
import sys

# subproject library
import modules.clsMail

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

''' Logging.
'''
def start_logging():
    """ Start logging of performance.
    """

    logging.captureWarnings(True)

    logger = logging.getLogger('DEV-TEST')

    formatter = logging.Formatter(' %(asctime)s     %(message)s \n', datefmt='%I:%M:%S %p')

    handler = logging.FileHandler('logging.test.txt', mode='w')

    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)

''' Auxiliary functions.
'''
def distribute_input(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    hours = args.hours
    notification = args.notification

    # Assertions
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    # Validity checks
    if notification:
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing
    return hours, notification

def finish(dict_, HOURS, notification):
    """ Finishing up a run of the testing battery.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Auxiliary objects
    hostname = socket.gethostname()

    # Finish logging
    with open('logging.test.txt', 'a') as file_:

        file_.write(' Summary \n\n')

        str_ = '   Test {0:<10} Success {1:<10} Failures  {2:<10}\n'

        for label in sorted(dict_.keys()):

            success = dict_[label]['success']

            failure = dict_[label]['failure']

            file_.write(str_.format(label, success, failure))

        file_.write('\n')

    # Send notification
    subject = ' GRMPY: Completed Testing Battery '

    message = ' A ' + str(HOURS) +' hour run of the testing battery on @' + hostname + ' is completed.'

    # Send notification
    if notification:

        mail_obj= modules.clsMail.MailCls()

        mail_obj.set_attr('subject', subject)

        mail_obj.set_attr('message', message)

        mail_obj.set_attr('attachment', 'logging.test.txt')

        mail_obj.lock()

        mail_obj.send()

def cleanup():
    """ Cleanup after test battery.
    """

    files = []

    files = files + glob.glob('*.grmpy.*')

    files = files + glob.glob('*.ini')

    files = files + glob.glob('*.pkl')

    files = files + glob.glob('*.txt')

    files = files + glob.glob('*.dat')

    for file_ in files:

        if 'logging' in file_:
            continue

        try:

            os.remove(file_)

        except OSError:

            pass

        try:

            shutil.rmtree(file_)

        except OSError:

            pass
