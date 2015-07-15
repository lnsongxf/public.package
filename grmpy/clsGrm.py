''' Module that holds all things related during the maximization setup.
'''

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class grmCls(metaCls):
    
    def __init__(self):
        
        self.attr = {}
        
        self.attr['modelObj']   = None
        
        self.attr['parasObj']   = None

        # Status.
        self.isLocked = False
        
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Model.
        assert (isinstance(self.attr['modelObj'], modelCls))
        assert (self.attr['modelObj'].getStatus() == True)
        
        # Parameters.
        assert (isinstance(self.attr['parasObj'], parasCls))
        assert (self.attr['parasObj'].getStatus() == True)
        
