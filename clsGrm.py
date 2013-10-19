''' Module that holds all things related during the maximization setup.
'''

# project library
import clsMeta
import clsModel
import clsParas
import clsRequest

class grmCls(clsMeta.meta):
    
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
        assert (isinstance(self.attr['modelObj'], clsModel.modelCls))
        assert (self.attr['modelObj'].getStatus() == True)
        
        # Parameters.
        assert (isinstance(self.attr['parasObj'], clsParas.parasCls))
        assert (self.attr['parasObj'].getStatus() == True)
        
        # Request.
        assert (isinstance(self.attr['requestObj'], clsRequest.requestCls))
        assert (self.attr['requestObj'].getStatus() == True)