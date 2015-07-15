''' This module contains the criterion function of the grmEstimatorToolbox.
'''
# standard library
import  numpy           as      np

from    scipy.stats     import  norm

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class critCls(metaCls):
    
    def __init__(self, model_obj, paras_obj):

        # Antibugging.
        assert (isinstance(model_obj, modelCls))
        assert (isinstance(paras_obj, parasCls))

        assert (model_obj.get_status() == True)
        assert (paras_obj.get_status() == True)

        self.attr = dict()
        
        self.attr['model_obj'] = model_obj

        self.attr['paras_obj'] = paras_obj

        # Status.
        self.isLocked = False
    
    def update(self, x):
        """ Update parameter object.
        """
        # Antibugging.
        assert (self.get_status() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
            
        # Distribute class attributes.
        paras_obj = self.getAttr('paras_obj')
        
        paras_obj.update(x, version='external', which='free')
    
    def evaluate(self, x, type_):
        """ Wrapper for function evaluate.
        """
        if type_ == 'function':
            rslt = self._evaluateFunction(x)
        elif type_ == 'gradient':
            rslt = self._evaluateGradient(x)

        # Finishing
        return rslt

    ''' Private methods for the calculation of the gradient.
    '''
    def _evaluateGradient(self, x):
        ''' Numerical approximation of gradient.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes.
        model_obj = self.getAttr('model_obj')

        paras_obj   = self.getAttr('paras_obj')
      
        epsilon     = model_obj.getAttr('epsilon')
        differences = model_obj.getAttr('differences')
         
        # Auxiliary statistics.
        numFree = paras_obj.getAttr('numFree')
        
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
        assert (self.get_status() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes.
        paras_obj = self.getAttr('paras_obj')

        model_obj = self.getAttr('model_obj')

        # Auxiliary objects
        version = model_obj.getAttr('version')

        # Update values.            
        self.update(x)

        # Likelihood calculation.
        if version == 'slow':

            likl = self._evaluateFunction_slow(paras_obj, model_obj)

        elif version == 'fast':

            likl = self._evaluateFunction_fast(paras_obj, model_obj)

        else:

            raise AssertionError

        # Transformations.
        likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

        # Quality checks.
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
         
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
