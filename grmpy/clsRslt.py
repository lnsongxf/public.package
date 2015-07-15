''' Module that contains the result class for the grmEstimatorToolbox.
'''

# standard library
import numpy    as np

import pickle as pkl

try:
   import cPickle as pkl
except:
   import pickle as pkl

import scipy
import copy
import random

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class rsltCls(metaCls):
    ''' This class contains all results provided back to the user from the 
        maximization setup.
    ''' 
    def __init__(self):
        
        # Attach attributes.
        self.attr = {}
        
        # Attributes.
        self.attr['grmObj']  = None
        self.attr['maxRslt'] = None
        self.attr['covMat']  = None
      
        self.attr['paraObjs'] = None
      
        # Constructed objects.
        self.attr['bmteExPost']    = None
        self.attr['cmteExAnte']    = None
        self.attr['smteExAnte']    = None        

        # Status indicator
        self.isLocked = False


    ''' Public methods
    '''
    def store(self, fileName):
        ''' Store class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isinstance(fileName, str))

        # Store.
        pkl.dump(self.attr, open(fileName, 'wb'))

    ''' Calculate derived attributes.
    '''
    def _derivedAttributes(self):
        ''' Construct derived objects.
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')
        covMat = self.getAttr('covMat')
        
        requestObj = grmObj.getAttr('requestObj')
        modelObj   = grmObj.getAttr('modelObj')
        parasObj   = grmObj.getAttr('parasObj')
       
        numAgents  = modelObj.getAttr('numAgents')
        
        alpha           = requestObj.getAttr('alpha')
        numDraws        = requestObj.getAttr('numDraws')
        withAsymptotics = requestObj.getAttr('withAsymptotics')

        # Auxiliary objects.
        parasCopy = copy.deepcopy(parasObj)
        
        paraObjs  = parasObj.getAttr('paraObjs')
        
        scale     = 1.0/numAgents
        cov       = scale*covMat
        
        # Sampling.
        np.random.seed(123), random.seed(456)
        
        externalValues   = parasObj.getValues(version = 'external', which = 'free')
         
        if(withAsymptotics):

            randomParameters = np.random.multivariate_normal(externalValues, cov, numDraws)
        
        else:
            
            randomParameters = np.zeros((len(externalValues), numDraws))
        
        ''' Core Structural Parameters.
        '''
        counter = 0
        
        for paraObj in paraObjs:
            
            if((paraObj.getAttr('isFree') == False) or (not withAsymptotics)):
                
                paraObj.setAttr('confi', ('---', '---'))

                paraObj.setAttr('pvalue', '---')
                                
            else:
                
                rslt = []
                
                for randomPara in randomParameters:
   
                    parasCopy.update(randomPara, version = 'external', which = 'free')
       
                    paraCopy = parasCopy.getParameter(counter)
       
                    rslt.append(paraCopy.getAttr('value'))           
                
                # Confidence intervals.
                lower, upper = scipy.stats.mstats.mquantiles(rslt, \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)])
                
                confi  = (lower, upper)

                paraObj.setAttr('confi', confi)

                # p values.
                estimate = paraObj.getAttr('value')
                
                pvalue = sum(np.sign(rslt) != np.sign(estimate))/float(numDraws)
                                
                paraObj.setAttr('pvalue', pvalue)
            
            counter += 1

        ''' Marginal Effects of Treatment. '''
        
        _ = self._addResultsMarginalEffects(randomParameters)
        
        ''' Store to file. '''
        
        _ = self._writeFile()
        
        ''' Store update parameter objects.'''
        
        self.attr['parasObj'] = parasObj

        self.attr['paras'] = parasObj.getValues('internal', 'all')

        # Cleanup.
        self.attr.pop('grmObj', None)
        self.attr.pop('parasObj', None)

    def _writeFile(self):
        ''' Write results to file.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Preparations
        grmObj     = self.getAttr('grmObj')
        requestObj = grmObj.getAttr('requestObj')
        parasObj   = grmObj.getAttr('parasObj')

        withAsymptotics         = requestObj.getAttr('withAsymptotics')
        surpEstimation          = parasObj.getAttr('surpEstimation')
        
        # Write results.
        with open('info.grmpy.out', 'a') as file_:
                
            self._writeMarginal(file_, withAsymptotics, surpEstimation)

    def _writeMarginal(self, file_, withAsymptotics, surpEstimation):
        ''' Write results on marginal effects of treatment.
        '''
        # Antibugging.
        assert (surpEstimation in [True, False])
        assert (withAsymptotics in [True, False])

         # Preparation
        struct = '''   {0[0]}        {0[1]}          {0[2]} / {0[3]}\n'''
        idx    = np.arange(0.01, 1.00, 0.01)
    
        parameterList = ['bmteExPost']
    
        if(surpEstimation): parameterList += ['cmteExAnte', 'smteExAnte']
        
        for parameter in parameterList:
            
            points = self.attr[parameter]['estimate']
            
            if(withAsymptotics):
                
                upperBound = self.attr[parameter]['confi']['upper']
    
                lowerBound = self.attr[parameter]['confi']['lower']
        
            if(parameter == 'bmteExPost'): 
                
                title = ' MARGINAL BENEFIT OF TREATMENT (EX POST)'
            
            if(parameter == 'cmteExAnte'): 
                
                title = ' MARGINAL COST OF TREATMENT '

            if(parameter == 'smteExAnte'): 
                
                title = ' MARGINAL SURPLUS OF TREATMENT '
                
            file_.write('\n' + title + '\n')
                       
            file_.write('\n' + '   Point     Estimate    Confidence Interval' + '\n\n')
               
            for i in range(99):
                
                u     = '{0:5.2f}'.format(idx[i])
                est   = '{0:5.2f}'.format(points[i])
                
                upper = '---'
                lower = '---'
                
                if(withAsymptotics):
                    
                    upper = '{0:5.2f}'.format(upperBound[i])
                    lower = '{0:5.2f}'.format(lowerBound[i])
                    
                file_.write(struct.format([u, est, lower, upper]))
                
    def _addResultsMarginalEffects(self, randomParameters):
        ''' Add results on marginal effects of treatment.
        '''
        
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isinstance(randomParameters, np.ndarray))
        assert (np.all(np.isfinite(randomParameters)))
        assert (randomParameters.dtype == 'float')
        assert (randomParameters.ndim  == 2)
    
        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')
        
        requestObj  = grmObj.getAttr('requestObj')
        modelObj    = grmObj.getAttr('modelObj')
        parasObj    = grmObj.getAttr('parasObj')
                
        withAsymptotics     = requestObj.getAttr('withAsymptotics')
        alpha               = requestObj.getAttr('alpha')

        surpEstimation      = parasObj.getAttr('surpEstimation')

        # Auxiliary objects.
        parasCopy = copy.deepcopy(parasObj)
        
        # Initialize parameters.
        parameterList = ['bmteExPost']

        if(surpEstimation): parameterList += ['smteExAnte', 'cmteExAnte']
        
        for parameter in parameterList:

            self.attr[parameter] = {}
            
            self.attr[parameter]['estimate'] = None
        
            self.attr[parameter]['confi']    = {}

        # Point estimates.
        args = {}
                
        for parameter in parameterList:
            
            args['which'] = parameter

            self.attr[parameter]['estimate'] = \
                self._construct_marginal_effects(modelObj, parasObj, args)
        
        # Confidence bounds.
        if(not withAsymptotics): return None
        
        rslt = {}
        
        args = {}
        
        for parameter in parameterList:
            
            args['which'] = parameter
            
            # Simulation.
            rslt[parameter] = []
            
            for randomPara in randomParameters:
   
                parasCopy.update(randomPara, version = 'external', which = 'free')
   
                rslt[parameter].append(
                    self._construct_marginal_effects(modelObj, parasCopy, args))
            
            # Type conversion.
            rslt[parameter] = np.array(rslt[parameter])
            
            # Confidence intervals.
            self.attr[parameter]['confi']['upper'] = []
            self.attr[parameter]['confi']['lower'] = []
               
            for i in range(99):
                
                lower, upper = scipy.stats.mstats.mquantiles(rslt[parameter][:,i], \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)], axis = 0)
            
                self.attr[parameter]['confi']['upper'].append(upper)
                self.attr[parameter]['confi']['lower'].append(lower)
                
        # Finishing.
        return None

    def _construct_marginal_effects(self, model_obj, paras_obj, args):
        """ Get effects.
        """
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isinstance(model_obj, modelCls))
        assert (model_obj.getStatus() == True)
        assert (isinstance(paras_obj, parasCls))
        assert (paras_obj.getStatus() == True)

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

    ''' Private methods.
    '''
    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        # Finishing.
        return True