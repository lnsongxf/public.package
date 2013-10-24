''' Module contains the class instance that manages all things related to the
    formulation of the user's request.
'''
# project library
import clsMeta

class requestCls(clsMeta.meta):
    ''' This class collects all information related to the user's request.
    '''
    def __init__(self):
        
        # Attach attributes.
        self.attr = {}
        
        # Optional arguments.
        self.attr['algorithm']       = None
        self.attr['epsilon']         = None
        self.attr['differences']     = None
        
        self.attr['gtol']            = None
        self.attr['maxiter']         = None

        self.attr['withAsymptotics'] = None
        self.attr['numDraws']        = None
        self.attr['hessian']         = None
        self.attr['numSims']         = None
        self.attr['alpha']           = None  

        self.attr['withMarginalEffects']     = None        
        self.attr['withConditionalEffects']  = None
        self.attr['withAverageEffects']      = None
                
        # Status
        self.isLocked = False
    
    ''' Private methods.
    '''
    def _checkIntegrity(self):
        ''' Check integrity of user request.
        '''       
        # withAsymptotics.
        assert (self.attr['withAsymptotics'] in [True, False])
        
        # Algorithm.
        assert (self.attr['algorithm'] in ['bfgs', 'powell'])

        # withConditionalAverageEffects. 
        assert (self.attr['withConditionalEffects'] in [True, False])   
        
        # withAverageEffects. 
        assert (self.attr['withAverageEffects'] in [True, False])       

        # withMarginalEffects. 
        assert (self.attr['withMarginalEffects'] in [True, False])  
            
        # Maximum iteration.
        if(self.attr['maxiter'] is not None):
            
            assert (isinstance(self.attr['maxiter'], int))
            assert (self.attr['maxiter'] >= 0)
        
        # alpha.
        assert (isinstance(self.attr['alpha'], float))
        assert (0.0 < self.attr['alpha'] < 1.0)

        # gtol.
        assert (isinstance(self.attr['gtol'], float))
        assert (self.attr['gtol'] > 0.00)
        
        # epsilon.
        assert (isinstance(self.attr['epsilon'], float))
        assert (self.attr['epsilon'] > 0.00)
        
        # differences.
        assert (self.attr['differences'] in ['one-sided', 'two-sided'])
                
        # hessian.
        assert (self.attr['hessian'] in ['bfgs', 'numdiff'])
        
        if(self.attr['algorithm'] == 'powell'):
                
            assert(self.attr['hessian'] == 'bfgs')
        
        # Finishing.
        return True
        