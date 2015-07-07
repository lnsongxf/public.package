''' Meta class for development test battery.
'''
# standard library
import logging
import socket
import glob
import os

# project library
from clsMail import mailCls


class meta(object):

    def __init__(self):
        ''' Initialization.
        '''

        pass

    def check(self, str_, err):
        ''' Check status.
        '''

        logger = logging.getLogger('MAIN')

        if(err):
            hostname = socket.gethostname()
            mailObj = mailCls()
            mailObj.setAttr('subject', ' grmToolbox: Failure in Testing Battery ')
            mailObj.setAttr('message', '\n There was a failed test case in the testing battery on @' + hostname + '.')
            mailObj.lock()
            mailObj.send()

            msg = '\n .. failure in test battery ' + str_ + ' ...\n'

            logger.info(msg)

            raise testingError(str_)

        else:

            msg = '\n .. successful test in battery ' + str_ + ' ...\n'

            logger.info(msg)

    def cleanup(self):
        ''' Basic cleanup operation.
        '''


        files = glob.glob('*.grm.*')

        for file_ in files:

            if(file_ == 'logging.grm.out'): continue

            try:

                os.remove(file_)

            except OSError:

                pass

class testingError(Exception):
    ''' Class that is confined to the definition of programmer exception
        type.
    '''

    def __init__(self, errmsg):

        self.errmsg = errmsg

    def __str__(self):

        return '\n\n errmsg: {0} \n'.format(self.errmsg)


