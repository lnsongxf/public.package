""" This module contains some custom exceptions for the continuous integration efforts.
"""

class TimedOutError(Exception):
    """ This custom error class allows to test whether an estimation can run for a limited amount of seconds without any complaints.
    """
    def __init__(self):

        self
