''' Module that holds all things related during the maximization setup.
'''

# project library
from grmpy.clsMeta import metaCls

import grmpy.clsModel
import grmpy.clsParas
import grmpy.clsRequest

class grmCls(metaCls):
    
    def __init__(self):
        
        self.attr = {}
        
        self.attr['modelObj']   = None
        
        self.attr['parasObj']   = None
        
        self.attr['requestObj'] = None
        
        # Status.
        self.isLocked = False
        
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Model.
        assert (isinstance(self.attr['modelObj'], grmpy.clsModel.modelCls))
        assert (self.attr['modelObj'].getStatus() == True)
        
        # Parameters.
        assert (isinstance(self.attr['parasObj'], grmpy.clsParas.parasCls))
        assert (self.attr['parasObj'].getStatus() == True)
        
        # Request.
        assert (isinstance(self.attr['requestObj'], grmpy.clsRequest.requestCls))
        assert (self.attr['requestObj'].getStatus() == True)
