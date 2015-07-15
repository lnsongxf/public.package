""" Module that contains the results class
"""

# standard library
try:
   import cPickle as pkl
except:
   import pickle as pkl

import numpy as np
import random
import scipy
import copy

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class RsltCls(metaCls):
    """ This class contains all results provided back to the user from the
        maximization setup.
    """
    def __init__(self):
        
        # Attach attributes.
        self.attr = dict()
        
        # Attributes.
        self.attr['paras_obj'] = None
        self.attr['model_obj'] = None

        self.attr['max_rslt'] = None
        self.attr['cov_mat'] = None
        self.attr['para_objs'] = None
      
        # Constructed objects.
        self.attr['bmte_ex_post'] = None
        self.attr['cmte_ex_ante'] = None
        self.attr['smte_ex_ante'] = None

        # Status indicator
        self.isLocked = False

    ''' Public methods
    '''
    def store(self, file_name):
        """ Store class instance.
        """
        # Antibugging.
        assert (self.get_status() == True)
        assert (isinstance(file_name, str))

        # Store.
        pkl.dump(self.attr, open(file_name, 'wb'))

    ''' Calculate derived attributes.
    '''
    def _derived_attributes(self):
        """ Construct derived objects.
        """
        # Antibugging.
        assert (self.get_status() == True)

        # Distribute class attributes.
        cov_mat = self.getAttr('cov_mat')
        
        model_obj = self.getAttr('modelObj')
        paras_obj = self.getAttr('parasObj')
       
        num_agents = model_obj.getAttr('numAgents')
        
        alpha = model_obj.getAttr('alpha')
        num_draws = model_obj.getAttr('numDraws')
        with_asymptotics = model_obj.getAttr('withAsymptotics')

        # Auxiliary objects.
        paras_copy = copy.deepcopy(paras_obj)
        
        para_objs  = paras_obj.getAttr('paraObjs')
        
        scale = 1.0 / num_agents
        cov = scale * cov_mat
        
        # Sampling.
        np.random.seed(123), random.seed(456)
        
        external_values = paras_obj.getValues(version='external', which='free')
         
        if with_asymptotics:

            random_parameters = np.random.multivariate_normal(external_values,
                                                              cov, num_draws)

        else:
            
            random_parameters = np.zeros((len(external_values), num_draws))
        
        ''' Core Structural Parameters.
        '''
        counter = 0
        
        for para_obj in para_objs:
            
            if (para_obj.getAttr('isFree') == False) or (not with_asymptotics):
                
                para_obj.setAttr('confi', ('---', '---'))

                para_obj.setAttr('pvalue', '---')
                                
            else:
                
                rslt = []
                
                for random_para in random_parameters:
   
                    paras_copy.update(random_para, version='external', which='free')
       
                    para_copy = paras_copy.getParameter(counter)
       
                    rslt.append(para_copy.getAttr('value'))
                
                # Confidence intervals.
                lower, upper = scipy.stats.mstats.mquantiles(rslt, \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)])
                
                confi  = (lower, upper)

                para_obj.setAttr('confi', confi)

                # p values.
                estimate = para_obj.getAttr('value')
                
                pvalue = sum(np.sign(rslt) != np.sign(estimate))/float(num_draws)
                                
                para_obj.setAttr('pvalue', pvalue)
            
            counter += 1

        ''' Marginal Effects of Treatment. '''
        
        self._add_results(random_parameters)
        
        ''' Store to file. '''
        
        self._write_file()
        
        ''' Store update parameter objects.'''
        
        self.attr['parasObj'] = paras_obj

        self.attr['paras'] = paras_obj.getValues('internal', 'all')

        # Cleanup.
        self.attr.pop('parasObj', None)

    def _write_file(self):
        """ Write results to file.
        """
        # Antibugging.
        assert (self.get_status() == True)
        
        # Preparations
        paras_obj = self.getAttr('paras_obj')
        model_obj = self.getAttr('model_obj')

        with_asymptotics = model_obj.getAttr('withAsymptotics')
        surp_estimation = paras_obj.getAttr('surpEstimation')
        
        # Write results.
        with open('info.grmpy.out', 'a') as file_:

            # Preparation
            struct = '''   {0[0]}        {0[1]}          {0[2]} / {0[3]}\n'''
            idx = np.arange(0.01, 1.00, 0.01)

            parameter_list = ['bmteExPost']

            if(surp_estimation): parameter_list += ['cmteExAnte', 'smteExAnte']

            for parameter in parameter_list:

                points = self.attr[parameter]['estimate']

                if(with_asymptotics):

                    upper_bound = self.attr[parameter]['confi']['upper']

                    lower_bound = self.attr[parameter]['confi']['lower']

                if parameter == 'bmteExPost':

                    title = ' MARGINAL BENEFIT OF TREATMENT (EX POST)'

                if parameter == 'cmteExAnte':

                    title = ' MARGINAL COST OF TREATMENT '

                if parameter == 'smteExAnte':

                    title = ' MARGINAL SURPLUS OF TREATMENT '

                file_.write('\n' + title + '\n')

                file_.write('\n' + '   Point     Estimate    Confidence Interval' + '\n\n')

                for i in range(99):

                    u = '{0:5.2f}'.format(idx[i])
                    est = '{0:5.2f}'.format(points[i])

                    upper = '---'
                    lower = '---'

                    if with_asymptotics:

                        upper = '{0:5.2f}'.format(upper_bound[i])
                        lower = '{0:5.2f}'.format(lower_bound[i])

                    file_.write(struct.format([u, est, lower, upper]))

    def _add_results(self, random_parameters):
        """ Add results on marginal effects of treatment.
        """
        # Antibugging.
        assert (self.get_status() == True)
        assert (isinstance(random_parameters, np.ndarray))
        assert (np.all(np.isfinite(random_parameters)))
        assert (random_parameters.dtype == 'float')
        assert (random_parameters.ndim  == 2)
    
        # Distribute class attributes.
        model_obj = self.getAttr('modelObj')
        paras_obj = self.getAttr('parasObj')
                
        with_asymptotics = model_obj.getAttr('withAsymptotics')
        alpha = model_obj.getAttr('alpha')
        surp_estimation = paras_obj.getAttr('surpEstimation')

        # Auxiliary objects.
        paras_copy = copy.deepcopy(paras_obj)
        
        # Initialize parameters.
        parameter_list = ['bmteExPost']

        if surp_estimation:
            parameter_list += ['smteExAnte', 'cmteExAnte']
        
        for parameter in parameter_list:
            self.attr[parameter] = {}
            self.attr[parameter]['estimate'] = None
            self.attr[parameter]['confi'] = {}

        # Point estimates.
        args = {}
                
        for parameter in parameter_list:
            
            args['which'] = parameter

            self.attr[parameter]['estimate'] = \
                self._construct_marginal_effects(model_obj, paras_obj, args)
        
        # Confidence bounds.
        if not with_asymptotics:
            return None
        
        rslt = {}
        
        args = {}
        
        for parameter in parameter_list:
            
            args['which'] = parameter
            
            # Simulation.
            rslt[parameter] = []
            
            for randomPara in random_parameters:
   
                paras_copy.update(randomPara, version = 'external', which = 'free')
   
                rslt[parameter].append(
                    self._construct_marginal_effects(model_obj, paras_copy, args))
            
            # Type conversion.
            rslt[parameter] = np.array(rslt[parameter])
            
            # Confidence intervals.
            self.attr[parameter]['confi']['upper'] = []
            self.attr[parameter]['confi']['lower'] = []
               
            for i in range(99):
                
                lower, upper = scipy.stats.mstats.mquantiles(rslt[parameter][:,i], \
                                prob=[(alpha*0.5), (1.0 - alpha*0.5)], axis = 0)
            
                self.attr[parameter]['confi']['upper'].append(upper)
                self.attr[parameter]['confi']['lower'].append(lower)
                
        # Finishing.
        return None

    def _construct_marginal_effects(self, model_obj, paras_obj, args):
        """ Get effects.
        """
        # Antibugging.
        assert (self.get_status() == True)
        assert (isinstance(model_obj, modelCls))
        assert (model_obj.get_status() == True)
        assert (isinstance(paras_obj, parasCls))
        assert (paras_obj.get_status() == True)

        # Distribute class attributes.
        x_ex_post_eval = model_obj.getAttr('xExPostEval')
        z_eval = model_obj.getAttr('zEval')
        c_eval = model_obj.getAttr('cEval')

        # Marginal benefit of treatment.
        rho_u1_v = paras_obj.getParameters('rho', 'U1,V')
        rho_u0_v = paras_obj.getParameters('rho', 'U0,V')

        coeffs_bene_ex_post = paras_obj.getParameters('bene', 'exPost')
        coeffs_cost = paras_obj.getParameters('cost', None)
        coeffs_choc = paras_obj.getParameters('choice', None)

        sd_v = paras_obj.getParameters('sd', 'V')
        sd_u1 = paras_obj.getParameters('sd', 'U1')
        sd_u0 = paras_obj.getParameters('sd', 'U0')

        bmte_level = np.dot(coeffs_bene_ex_post, x_ex_post_eval)
        smte_level = np.dot(coeffs_choc, z_eval)
        cmte_level = np.dot(coeffs_cost, c_eval)

        bmte_ex_post = np.tile(np.nan, 99)
        cmte_ex_post = np.tile(np.nan, 99)
        smte_ex_ante = np.tile(np.nan, 99)

        eval_points = np.round(np.arange(0.01, 1.0, 0.01), decimals=2)
        quantiles = scipy.stats.norm.ppf(eval_points, loc=0, scale=sd_v)

        # Construct marginal benefit of treatment (ex post)
        bmte_slopes = ((sd_u1/sd_v)*rho_u1_v - (sd_u0/sd_v)*rho_u0_v)*quantiles
        smte_slopes = -quantiles
        cmte_slopes = (((sd_u1/sd_v)*rho_u1_v - (sd_u0/sd_v)*rho_u0_v) +
                       1.0)*quantiles

        # Construct marginal benefit of treatment (ex post)
        for i in range(99):

            bmte_ex_post[i] = bmte_level + bmte_slopes[i]

        # Construct marginal surplus of treatment (ex ante).
        for i in range(99):

            smte_ex_ante[i] = smte_level + smte_slopes[i]

        # Construct marginal cost of treatment (ex ante).
        for i in range(99):

            cmte_ex_post[i] = cmte_level + cmte_slopes[i]

        if args['which'] == 'bmteExPost':

            rslt = bmte_ex_post

        elif args['which'] == 'cmteExAnte':

            rslt = cmte_ex_post

        elif args['which'] == 'smteExAnte':

            rslt = smte_ex_ante

        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (99, ))

        # Finishing.
        return rslt
