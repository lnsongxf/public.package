""" Module that contains the results class
"""

# standard library
import pickle as pkl
import numpy as np
import random
import scipy
import copy

# project library
from grmpy.clsMeta import MetaCls
from grmpy.clsModel import ModelCls
from grmpy.clsParas import ParasCls

class RsltCls(MetaCls):
    """ This class contains all results provided back to the user from the
        maximization setup.
    """
    def __init__(self, model_obj, paras_obj):

        # Antibugging
        assert (isinstance(model_obj, ModelCls))
        assert (isinstance(paras_obj, ParasCls))

        assert (model_obj.get_status() is True)
        assert (paras_obj.get_status() is True)

        # Attributes.
        self.attr = dict()
        self.attr['model_obj'] = model_obj
        self.attr['paras_obj'] = paras_obj

        self.attr['max_rslt'] = None
        self.attr['cov_mat'] = None

        # Constructed objects.
        self.attr['bmte_ex_post'] = None
        self.attr['cmte_ex_ante'] = None
        self.attr['smte_ex_ante'] = None

        # Status indicator
        self.is_locked = False

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
    def derived_attributes(self):
        """ Construct derived objects.
        """
        # Antibugging.
        assert (self.get_status() == True)

        # Distribute class attributes.
        cov_mat = self.get_attr('cov_mat')
        
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')
       
        num_agents = model_obj.get_attr('num_agents')
        
        alpha = model_obj.get_attr('alpha')
        num_draws = model_obj.get_attr('numDraws')
        with_asymptotics = model_obj.get_attr('with_asymptotics')

        # Auxiliary objects.
        paras_copy = copy.deepcopy(paras_obj)
        
        para_objs  = paras_obj.get_attr('para_objs')
        
        scale = 1.0 / num_agents
        cov = scale * cov_mat
        
        # Sampling.
        np.random.seed(123), random.seed(456)
        
        external_values = paras_obj.get_values(version='external', which='free')
         
        if with_asymptotics:

            random_parameters = np.random.multivariate_normal(external_values,
                                                              cov, num_draws)

        else:
            
            random_parameters = np.zeros((len(external_values), num_draws))
        
        ''' Core Structural Parameters.
        '''
        counter = 0
        
        for para_obj in para_objs:
            
            if (para_obj.get_attr('is_free') == False) or (not with_asymptotics):
                
                para_obj.set_attr('confi', ('---', '---'))

                para_obj.set_attr('pvalue', '---')
                                
            else:
                
                rslt = []
                
                for random_para in random_parameters:
   
                    paras_copy.update(random_para, version='external', which='free')
       
                    para_copy = paras_copy.get_parameter(counter)
       
                    rslt.append(para_copy.get_attr('value'))
                
                # Confidence intervals.
                lower, upper = scipy.stats.mstats.mquantiles(rslt, \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)])
                
                confi  = (lower, upper)

                para_obj.set_attr('confi', confi)

                # p values.
                estimate = para_obj.get_attr('value')
                
                pvalue = sum(np.sign(rslt) != np.sign(estimate))/float(num_draws)
                                
                para_obj.set_attr('pvalue', pvalue)
            
            counter += 1

        ''' Marginal Effects of Treatment. '''
        
        self._add_results(random_parameters)
        
        ''' Store to file. '''
        
        self._write_file()
        
        ''' Store update parameter objects.'''
        
        self.attr['paras_obj'] = paras_obj

        self.attr['paras'] = paras_obj.get_values('internal', 'all')

        # Cleanup.
        self.attr.pop('paras_obj', None)

    def _write_file(self):
        """ Write results to file.
        """
        # Antibugging.
        assert (self.get_status() == True)
        
        # Preparations
        paras_obj = self.get_attr('paras_obj')
        model_obj = self.get_attr('model_obj')

        with_asymptotics = model_obj.get_attr('with_asymptotics')
        surp_estimation = paras_obj.get_attr('surp_estimation')
        
        # Write results.
        with open('info.grmpy.out', 'a') as file_:

            # Preparation
            struct = '''   {0[0]}        {0[1]}          {0[2]} / {0[3]}\n'''
            idx = np.arange(0.01, 1.00, 0.01)

            parameter_list = ['bmte_ex_post']

            if(surp_estimation): parameter_list += ['cmte_ex_ante', 'smte_ex_ante']

            for parameter in parameter_list:

                points = self.attr[parameter]['estimate']

                if(with_asymptotics):

                    upper_bound = self.attr[parameter]['confi']['upper']

                    lower_bound = self.attr[parameter]['confi']['lower']

                if parameter == 'bmte_ex_post':

                    title = ' MARGINAL BENEFIT OF TREATMENT (EX POST)'

                if parameter == 'cmte_ex_ante':

                    title = ' MARGINAL COST OF TREATMENT '

                if parameter == 'smte_ex_ante':

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
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')
                
        with_asymptotics = model_obj.get_attr('with_asymptotics')
        alpha = model_obj.get_attr('alpha')
        surp_estimation = paras_obj.get_attr('surp_estimation')

        # Auxiliary objects.
        paras_copy = copy.deepcopy(paras_obj)
        
        # Initialize parameters.
        parameter_list = ['bmte_ex_post']

        if surp_estimation:
            parameter_list += ['smte_ex_ante', 'cmte_ex_ante']
        
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
        assert (isinstance(model_obj, ModelCls))
        assert (model_obj.get_status() == True)
        assert (isinstance(paras_obj, ParasCls))
        assert (paras_obj.get_status() == True)

        # Distribute class attributes.
        x_ex_post_eval = model_obj.get_attr('x_ex_post_eval')
        z_eval = model_obj.get_attr('z_eval')
        c_eval = model_obj.get_attr('c_eval')

        # Marginal benefit of treatment.
        rho_u1_v = paras_obj.get_parameters('rho', 'U1,V')
        rho_u0_v = paras_obj.get_parameters('rho', 'U0,V')

        coeffs_bene_ex_post = paras_obj.get_parameters('bene', 'exPost')
        coeffs_cost = paras_obj.get_parameters('cost', None)
        coeffs_choc = paras_obj.get_parameters('choice', None)

        sd_v = paras_obj.get_parameters('sd', 'V')
        sd_u1 = paras_obj.get_parameters('sd', 'U1')
        sd_u0 = paras_obj.get_parameters('sd', 'U0')

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

        if args['which'] == 'bmte_ex_post':

            rslt = bmte_ex_post

        elif args['which'] == 'cmte_ex_ante':

            rslt = cmte_ex_post

        elif args['which'] == 'smte_ex_ante':

            rslt = smte_ex_ante

        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (99, ))

        # Finishing.
        return rslt
