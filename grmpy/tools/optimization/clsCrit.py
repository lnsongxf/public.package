""" This module contains the criterion function of the GRMPY package.
"""
# standard library
from scipy.stats import norm
import numpy as np

# project library
from grmpy.clsMeta import MetaCls
from grmpy.clsModel import ModelCls
from grmpy.clsParas import parasCls

class CritCls(MetaCls):
    
    def __init__(self, model_obj, paras_obj):

        # Antibugging
        assert (isinstance(model_obj, ModelCls))
        assert (isinstance(paras_obj, parasCls))

        assert (model_obj.get_status() is True)
        assert (paras_obj.get_status() is True)

        self.attr = dict()
        
        self.attr['model_obj'] = model_obj

        self.attr['paras_obj'] = paras_obj

        # Status
        self.is_locked = False
    
    def update(self, x):
        """ Update parameter object.
        """
        # Antibugging
        assert (self.get_status() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
            
        # Distribute class attributes
        paras_obj = self.get_attr('paras_obj')

        # Update parameter object
        paras_obj.update(x, 'external', 'free')
    
    def evaluate(self, x, type_):
        """ Wrapper for function evaluate.
        """
        if type_ == 'function':
            rslt = self._evaluate_function(x)
        elif type_ == 'gradient':
            rslt = self._evaluate_gradient(x)

        # Finishing
        return rslt

    ''' Private methods for the calculation of the gradient.
    '''
    def _evaluate_gradient(self, x):
        """ Numerical approximation of gradient.
        """
        # Antibugging
        assert (self.get_status() is True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')

        epsilon = model_obj.get_attr('epsilon')
        differences = model_obj.get_attr('differences')
         
        # Auxiliary statistics
        num_free = paras_obj.get_attr('numFree')
        
        # Antibugging.
        assert (x.shape == (num_free, ))
        
        # Calculate baseline
        f0 = self.evaluate(x, 'function')
        
        # Initialization
        grad = np.zeros(num_free, dtype='float')
        ei = np.zeros(num_free, dtype='float')
    
        # Gradient loop
        for k in range(num_free):
            
            # Calculate step size
            ei[k] = 1.0
            d = epsilon*ei
    
            # Gradient approximation.
            if differences == 'one-sided':
                upper = self._evaluate_function(x + d,  False)
                lower = f0
                grad[k] = (upper - lower)/d[k]
            
            if differences == 'two-sided':
                upper = self._evaluate_function(x + d, False)
                lower = self._evaluate_function(x - d, False)
                grad[k] = (upper - lower)/(2.0*d[k])
            
            # Reset step size
            ei[k] = 0.0
        
        # Check quality
        assert (isinstance(grad, np.ndarray))
        assert (np.all(np.isfinite(grad)))
        assert (grad.shape == (num_free, ))
        assert (grad.dtype == 'float')
                
        # Finishing
        return grad
    
    def _evaluate_function(self, x, logging = True):
        """ Negative log-likelihood function of the grmEstimatorToolbox.
        """
        # Antibugging
        assert (self.get_status() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.ndim == 1)
        
        # Distribute class attributes
        paras_obj = self.get_attr('paras_obj')
        model_obj = self.get_attr('model_obj')

        # Auxiliary objects
        version = model_obj.get_attr('version')

        # Update values
        self.update(x)

        # Likelihood calculation
        if version == 'slow':
            likl = self._evaluate_function_slow(paras_obj, model_obj)

        elif version == 'fast':
            likl = self._evaluate_function_fast(paras_obj, model_obj)

        else:
            raise AssertionError

        # Transformations
        likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

        # Quality checks
        assert (isinstance(likl, float))    
        assert (np.isfinite(likl))
         
        # Finishing
        return likl    

    ''' Private class attributes.
    '''
    @staticmethod
    def _evaluate_function_fast(paras_obj, model_obj):
        """ Evaluate the criterion function in a fast fashion.
        """
        # Distribute model information
        x_ex_post = model_obj.get_attr('X_ex_post')
        y = model_obj.get_attr('Y')
        d = model_obj.get_attr('D')
        z = model_obj.get_attr('Z')

        # Distribute current parametrization
        outc_treated = paras_obj.getParameters('outc', 'treated')
        outc_untreated = paras_obj.getParameters('outc', 'untreated')
        coeffs_choc = paras_obj.getParameters('choice', None)

        sd_u1 = paras_obj.getParameters('sd',  'U1')
        sd_u0 = paras_obj.getParameters('sd',  'U0')
        sd_v = paras_obj.getParameters('sd',  'V')
        var_v = paras_obj.getParameters('var',  'V')

        rho_u1_v = paras_obj.getParameters('rho', 'U1,V')
        rho_u0_v = paras_obj.getParameters('rho', 'U0,V')

        # Construct choice index
        choice_indices = np.dot(coeffs_choc, z.T)

        # Calculate densities
        arg_one = d*(y - np.dot(outc_treated, x_ex_post.T))/sd_u1 + \
            (1 - d)*(y - np.dot(outc_untreated, x_ex_post.T))/sd_u0
        arg_two = d*(choice_indices - sd_v*rho_u1_v*arg_one)/np.sqrt((1.0 -
            rho_u1_v**2)*var_v) + (1 - d)*(choice_indices - \
            sd_v*rho_u0_v*arg_one)/np.sqrt((1.0 - rho_u0_v**2)*var_v)

        # Evaluate densities
        cdf_evals, pdf_evals = norm.cdf(arg_two), norm.pdf(arg_one)

        # Calculate individual likelihoods
        likl = d*(1.0/sd_u1) * pdf_evals * cdf_evals + \
            (1 - d) * (1.0/sd_u0) * pdf_evals * (1.0 - cdf_evals)

        # Finishing
        return likl

    @staticmethod
    def _evaluate_function_slow(paras_obj, model_obj):
        """ Evaluate the criterion function in a slow fashion.
        """
        # Distribute model information
        num_agents = model_obj.get_attr('num_agents')
        x_ex_post = model_obj.get_attr('X_ex_post')
        y = model_obj.get_attr('Y')
        d = model_obj.get_attr('D')
        z = model_obj.get_attr('Z')

        # Distribute current parametrization
        outc_treated = paras_obj.getParameters('outc', 'treated')
        outc_untreated = paras_obj.getParameters('outc', 'untreated')
        coeffs_choc = paras_obj.getParameters('choice', None)

        sd_u1 = paras_obj.getParameters('sd', 'U1')
        sd_u0 = paras_obj.getParameters('sd', 'U0')
        sd_v = paras_obj.getParameters('sd', 'V')

        rho_u1_v = paras_obj.getParameters('rho', 'U1,V')
        rho_u0_v = paras_obj.getParameters('rho', 'U0,V')

        # Initialize containers
        likl = np.tile(np.nan, num_agents)
        choice_idx = np.tile(np.nan, num_agents)

        for i in range(num_agents):

            # Construct choice index
            choice_idx[i] = np.dot(coeffs_choc, z[i,:])

            # Select outcome information
            if d[i] == 1.00:
                coeffs, rho, sd = outc_treated, rho_u1_v, sd_u1
            else:
                coeffs, rho, sd = outc_untreated, rho_u0_v, sd_u0

            # Calculate densities
            arg_one = (y[i] - np.dot(coeffs, x_ex_post[i, :])) / sd
            arg_two = (choice_idx[i] - rho * sd_v * arg_one) / \
                np.sqrt((1.0 - rho ** 2) * sd_v**2)

            pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

            # Construct likelihood
            if d[i] == 1.0:
                contrib = (1.0 / float(sd)) * pdf_evals * cdf_evals
            else:
                contrib = (1.0 / float(sd)) * pdf_evals * (1.0 - cdf_evals)

            # Collect individual information
            likl[i] = contrib

        # Finishing
        return likl
