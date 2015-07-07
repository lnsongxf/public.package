''' Auxiliary functions for development test suite.
'''

__all__ = ['distributeInput', 'finish', 'cleanup', \
           'startLogging']

# standard library
import logging
import socket
import shutil
import glob
import os


# subproject library
import modules.clsMail

''' Logging.
'''
def startLogging():
    ''' Start logging of performance.
    '''

    logging.captureWarnings(True)

    logger    = logging.getLogger('DEV-TEST')

    formatter = logging.Formatter(' %(asctime)s     %(message)s \n', datefmt = '%I:%M:%S %p')

    handler   = logging.FileHandler('logging.test.txt', mode = 'w')

    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)

''' Auxiliary functions.
'''
def distributeInput(parser):
    ''' Check input for estimation script.
    '''
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    hours = args.hours
    notification = args.notification

    # Assertions.
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    # Validity checks
    if notification:
        # Check that the credentials file is stored in the user's HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing.
    return hours, notification

def finish(dict_, HOURS, notification):
    ''' Finishing up a run of the testing battery.
    '''
    # Antibugging.
    assert (isinstance(dict_, dict))

    # Auxiliary objects.
    hostname = socket.gethostname()

    # Finish logging.
    with open('logging.test.txt', 'a') as file_:

        file_.write(' Summary \n\n')

        str_ = '   Test {0:<10} Success {1:<10} Failures  {2:<10}\n'

        for label in sorted(dict_.keys()):

            success = dict_[label]['success']

            failure = dict_[label]['failure']

            file_.write(str_.format(label, success, failure))

        file_.write('\n')

    # Send notification.
    subject = ' GRMPY: Completed Testing Battery '

    message = ' A ' + str(HOURS) +' hour run of the testing battery on @' + hostname + ' is completed.'


    if notification:

        mailObj = modules.clsMail.mailCls()

        mailObj.setAttr('subject', subject)

        mailObj.setAttr('message', message)

        mailObj.setAttr('attachment', 'logging.test.txt')

        mailObj.lock()

        mailObj.send()

def cleanup():
    ''' Cleanup after test battery.
    '''

    files = []

    files = files + glob.glob('*.grm.*')

    files = files + glob.glob('.grm.*')

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