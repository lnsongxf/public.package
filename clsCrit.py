''' This module contains the criterion function of the grmEstimatorToolbox.
'''
# standard library
import  numpy           as      np

from    scipy.stats     import  norm

# project library
import clsMeta
import clsGrm

class critCls(clsMeta.meta):
    
    def __init__(self, grmObj):

        # Antibugging.
        assert (isinstance(grmObj, clsGrm.grmCls))
        assert (grmObj.getStatus() == True)

        self.attr = {}
        
        self.attr['grmObj'] = grmObj

        # Results.                
        self.attr['currentFval'] = None
        
        self.attr['startFval']   = None
        
        self.attr['stepFval']    = None          
        
        self.attr['numStep']     = 0

        # Status.
        self.isLocked = False
    
    def update(self, x):
        ''' Update parameter object.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
            
        # Distribute class attributes.
        grmObj   = self.getAttr('grmObj')
        
        parasObj = grmObj.getAttr('parasObj')        
        
        parasObj.updateValues(x, isExternal = True, isAll = False)
    
    def evaluate(self, x, type_):
        ''' Wrapper for function evaluate.
        '''
        
        if(type_ == 'function'):
            
            rslt = self._evaluateFunction(x)
            
        elif(type_ == 'gradient'):
            
            rslt = self._evaluateGradient(x)
            
        return rslt

    ''' Private methods for the calculation of the gradient.
    '''
    def _evaluateGradient(self, x):
        ''' Numerical approximation of gradient.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')
        
        requestObj = grmObj.getAttr('requestObj')

        parasObj   = grmObj.getAttr('parasObj')
      
        epsilon     = requestObj.getAttr('epsilon')
        differences = requestObj.getAttr('differences')
        
        maxiter     = requestObj.getAttr('maxiter')
        
        # Auxiliary statistics.
        numFree = parasObj.getAttr('numFree')
        
        # Applicability.
        if(maxiter == 0):
            
            return np.zeros(numFree)
        
        # Antibugging.
        assert (x.shape == (numFree, ))
        
        # Calculate baseline.
        f0 = self.evaluate(x, 'function')
        
        # Initialization.
        grad = np.zeros(numFree, dtype = 'float')
        ei   = np.zeros(numFree, dtype = 'float')
    
        # Gradient loop.
        for k in range(numFree):
            
            # Calculate step size.
            ei[k] = 1.0
            
            d     = epsilon*ei
    
            # Gradient approximation.
            if(differences == 'one-sided'):
                
                upper = self.evaluate(x + d, 'function')
                
                lower = f0
                
                
                grad[k] = (upper - lower)/d[k]
            
            if(differences == 'two-sided'):
                
                upper = self.evaluate(x + d, 'function')
                
                lower = self.evaluate(x - d, 'function')
                
                
                grad[k] = (upper - lower)/(2.0*d[k])
            
            # Reset step size.
            ei[k] = 0.0
        
        # Check quality.
        assert (isinstance(grad, np.ndarray))
        assert (np.all(np.isfinite(grad)))
        assert (grad.shape == (numFree, ))
        assert (grad.dtype == 'float')
                
        # Finishing.
        return grad
    
    def _evaluateFunction(self, x):
        ''' Negative log-likelihood function of the grmEstimatorToolbox.
        '''    
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')
        
        modelObj = grmObj.getAttr('modelObj')
        
        parasObj = grmObj.getAttr('parasObj')
        
        
        numAgents = modelObj.getAttr('numAgents')
        
        xExPost   = modelObj.getAttr('xExPost')    
        
        Y         = modelObj.getAttr('Y')
        
        D         = modelObj.getAttr('D')
        
        Z         = modelObj.getAttr('Z')
                
        # Update values.            
        self.update(x)

        # Distribute current parametrization.
        outcTreated   = parasObj.getParameters('outc', 'treated')
        outcUntreated = parasObj.getParameters('outc', 'untreated') 
        coeffsChoc    = parasObj.getParameters('choice', None)
        
        sdU1    = parasObj.getParameters('sd',  'U1') 
        sdU0    = parasObj.getParameters('sd',  'U0') 
        sdV     = parasObj.getParameters('sd',  'V')  
        varV    = parasObj.getParameters('var', 'V') 
            
        rhoU1V  = parasObj.getParameters('rho', 'U1,V')  
        rhoU0V  = parasObj.getParameters('rho', 'U0,V')  
        
        # Likelihood calculation.
        choiceIndices = np.dot(coeffsChoc, Z.T) 
    
        argOne = D*(Y - np.dot(outcTreated, xExPost.T))/sdU1 + \
                (1 - D)*(Y - np.dot(outcUntreated, xExPost.T))/sdU0
    
        argTwo = D*(choiceIndices - sdV*rhoU1V*argOne)/np.sqrt((1.0 - rhoU1V**2)*varV) + \
                (1 - D)*(choiceIndices - sdV*rhoU0V*argOne)/np.sqrt((1.0 - rhoU0V**2)*varV)
        
        cdfEvals = norm.cdf(argTwo)
        pdfEvals = norm.pdf(argOne)
    
        likl = D*(1.0/sdU1)*pdfEvals*cdfEvals + \
                    (1 - D)*(1.0/sdU0)*pdfEvals*(1.0  - cdfEvals)
    
        # Transformations.
        likl = np.clip(likl, 1e-20, np.inf)
        
        likl = -np.log(likl)
        
        likl = likl.sum()
        
        likl = (1.0/float(numAgents))*likl
    
        # Quality checks.
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
        assert (likl > 0.0)
        
        # Logging.
        self._logging(likl)
    
        #Finishing.        
        return likl    

    ''' Private class attributes.
    '''
    def _logging(self, likl):
        ''' Logging of progress.
        '''
        # Antibugging.
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
        assert (likl > 0.0)
        
        # Logging.
        self.attr['currentFval'] = likl
                
        isStart = (self.attr['startFval'] is None)

        if(isStart):
            
            self.attr['startFval'] = self.attr['currentFval']
            self.attr['stepFval']  = self.attr['currentFval']
        
            self._write('start')
        
        isStep = (self.attr['currentFval'] < self.attr['stepFval'])
            
        if(isStep or isStart):

            self.attr['stepFval'] = self.attr['currentFval']
            
            self._write('step')
        
            file_ = open('grmEstimator.grm.log', 'a')
            
            if(isStart): 
                
                file_.write('  Start ')

            else:
            
                file_.write('  Step ' + str(self.attr['numStep']))
            
            file_.write('\n\n')
            file_.write('    Function Value: ' + str(self.attr['stepFval']))
            file_.write('\n\n\n')
            
            file_.close()       
            
            self.attr['numStep']   += 1

    def _write(self, task):
        ''' Write information to disk.
        '''
        # Antibugging.
        assert (task in ['start', 'step'])
        
        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')

        parasObj = grmObj.getAttr('parasObj')

        # Collect objects.
        paras = parasObj.getValues(isExternal = False, isAll = False)
        
        # Save.
        np.savetxt(task + 'Paras.grm.out', paras, fmt = '%25.12f')
        
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # GrmObject.
        assert (isinstance(self.attr['grmObj'], clsGrm.grmCls))
        assert (self.attr['grmObj'].getStatus() == True)

        # Function values.
        for type_ in ['currentFval', 'startFval', 'stepFval']:
            
            if(self.attr[type_] is not None):
                
                assert (isinstance(self.attr[type_], float))
        
        # Number of steps.
        assert (isinstance(self.attr['numStep'], int))
        assert (self.attr['numStep'] >= 0)