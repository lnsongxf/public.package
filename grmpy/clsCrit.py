''' This module contains the criterion function of the grmEstimatorToolbox.
'''
# standard library
import  numpy           as      np

from    scipy.stats     import  norm

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsGrm import grmCls

class critCls(metaCls):
    
    def __init__(self, grmObj):

        # Antibugging.
        assert (isinstance(grmObj, grmCls))
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
        
        parasObj.update(x, version = 'external', which = 'free')
    
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
         
        # Auxiliary statistics.
        numFree = parasObj.getAttr('numFree')
        
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
                
                upper = self._evaluateFunction(x + d,  False)
                
                lower = f0
                
                
                grad[k] = (upper - lower)/d[k]
            
            if(differences == 'two-sided'):
                
                upper = self._evaluateFunction(x + d, False)
                
                lower = self._evaluateFunction(x - d, False)
                
                
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
    
    def _evaluateFunction(self, x, logging = True):
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

        requestObj = grmObj.getAttr('requestObj')

        # Auxiliary objects
        version = requestObj.getAttr('version')

        # Update values.            
        self.update(x)

        # Likelihood calculation.
        if version == 'slow':

            likl = self._evaluateFunction_slow(parasObj, modelObj)

        elif version == 'fast':

            likl = self._evaluateFunction_fast(parasObj, modelObj)

        else:

            raise AssertionError

        # Transformations.
        likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

        # Quality checks.
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
         
        # Logging.
        if(logging): self._logging(likl)
    
        #Finishing.        
        return likl    

    ''' Private class attributes.
    '''
    def _evaluateFunction_fast(self, parasObj, modelObj):
        """ Evaluate the criterion function in a fast fashion.
        """
        # Distribute model information
        xExPost = modelObj.getAttr('xExPost')

        Y = modelObj.getAttr('Y')

        D = modelObj.getAttr('D')

        Z = modelObj.getAttr('Z')

        # Distribute current parametrization.
        outcTreated = parasObj.getParameters('outc', 'treated')
        outcUntreated = parasObj.getParameters('outc', 'untreated')
        coeffsChoc = parasObj.getParameters('choice', None)

        sdU1 = parasObj.getParameters('sd',  'U1')
        sdU0 = parasObj.getParameters('sd',  'U0')
        sdV = parasObj.getParameters('sd',  'V')
        varV = parasObj.getParameters('var',  'V')

        rhoU1V = parasObj.getParameters('rho', 'U1,V')
        rhoU0V = parasObj.getParameters('rho', 'U0,V')

        # Construct choice index
        choiceIndices = np.dot(coeffsChoc, Z.T)

        # Calculate densities
        argOne = D*(Y - np.dot(outcTreated, xExPost.T))/sdU1 + \
            (1 - D)*(Y - np.dot(outcUntreated, xExPost.T))/sdU0
        argTwo = D*(choiceIndices - sdV*rhoU1V*argOne)/np.sqrt((1.0 - rhoU1V**2)*varV) + \
            (1 - D)*(choiceIndices - sdV*rhoU0V*argOne)/np.sqrt((1.0 - rhoU0V**2)*varV)

        # Evaluate densities
        cdfEvals, pdfEvals = norm.cdf(argTwo), norm.pdf(argOne)

        # Calculate individual likelihoods
        likl = D*(1.0/sdU1)*pdfEvals*cdfEvals + \
            (1 - D)*(1.0/sdU0)*pdfEvals*(1.0  - cdfEvals)

        # Finishing
        return likl

    def _evaluateFunction_slow(self, parasObj, modelObj):
        """ Evaluate the criterion function in a slow fashion.
        """
        # Distribute model information
        numAgents = modelObj.getAttr('numAgents')

        xExPost = modelObj.getAttr('xExPost')

        Y = modelObj.getAttr('Y')

        D = modelObj.getAttr('D')

        Z = modelObj.getAttr('Z')

        # Distribute current parametrization.
        outcTreated = parasObj.getParameters('outc', 'treated')
        outcUntreated = parasObj.getParameters('outc', 'untreated')
        coeffsChoc = parasObj.getParameters('choice', None)

        sdU1 = parasObj.getParameters('sd',  'U1')
        sdU0 = parasObj.getParameters('sd',  'U0')
        sdV = parasObj.getParameters('sd',  'V')

        rhoU1V = parasObj.getParameters('rho', 'U1,V')
        rhoU0V = parasObj.getParameters('rho', 'U0,V')

        # Initialize containers
        likl = np.tile(np.nan, numAgents)
        choice_idx = np.tile(np.nan, numAgents)

        for i in range(numAgents):

            # Construct choice index
            choice_idx[i] = np.dot(coeffsChoc, Z[i,:])

            # Select outcome information
            if D[i] == 1.00:
                coeffs, rho, sd = outcTreated, rhoU1V, sdU1
            else:
                coeffs, rho, sd = outcUntreated, rhoU0V, sdU0

            # Calculate densities
            arg_one = (Y[i] - np.dot(coeffs, xExPost[i, :])) / sd
            arg_two = (choice_idx[i] - rho * sdV * arg_one) / \
                np.sqrt((1.0 - rho ** 2) * sdV**2)

            pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

            # Construct likelihood
            if D[i] == 1.0:
                contrib = (1.0 / float(sd)) * pdf_evals * cdf_evals
            else:
                contrib = (1.0 / float(sd)) * pdf_evals * (1.0 - cdf_evals)

            # Collect individual information
            likl[i] = contrib

        # Finishing
        return likl

    def _logging(self, likl):
        ''' Logging of progress.
        '''
        # Antibugging.
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
   
        # Distribute class attributes.
        grmObj    = self.attr['grmObj']
        
        modelObj  = grmObj.getAttr('modelObj')
        
        numAgents = modelObj.getAttr('numAgents')
        
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
        
            file_ = open('grmToolbox.grm.log', 'a')
            
            if(isStart): 
                
                file_.write('\n Estimation Sample: ' + str(numAgents) + '\n')
                
                file_.write('\n  Start ')

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
        paras = parasObj.getValues(version = 'internal', which = 'all')
        
        # Save.
        np.savetxt(task + 'Paras.grm.out', paras, fmt = '%25.12f')
        
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # GrmObject.
        assert (isinstance(self.attr['grmObj'], grmCls))
        assert (self.attr['grmObj'].getStatus() == True)

        # Function values.
        for type_ in ['currentFval', 'startFval', 'stepFval']:
            
            if(self.attr[type_] is not None):
                
                assert (isinstance(self.attr[type_], float))
        
        # Number of steps.
        assert (isinstance(self.attr['numStep'], int))
        assert (self.attr['numStep'] >= 0)