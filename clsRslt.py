''' Module that contains the result class for the grmEstimatorToolbox.
'''

# standard library
import numpy    as np

import scipy
import copy
import random

# project library
import clsMeta
import clsEffects

class results(clsMeta.meta):
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

        self.attr['bteExPost']     = None
        self.attr['bteExAnte']     = None
        
        self.attr['cte']           = None      
        self.attr['ste']           = None      

        self.attr['bteExPostCond'] = None
        self.attr['bteExAnteCond'] = None
        
        self.attr['cteCond']       = None      
        self.attr['steCond']       = None      

        # Status indicator
        self.isLocked = False
        
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
        np.random.seed(123)
        random.seed(456)
        
        externalValues   = parasObj.getValues(isExternal = True, isAll = False)
         
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
   
                    parasCopy.updateValues(randomPara, isExternal = True, isAll = False)
       
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
        
        ''' Unconditional Average Effects of Treatment. '''
        
        _ = self._addResultsAverageEffects(randomParameters)

        ''' Conditional Average Effects of Treatment. '''
        
        _ = self._addResultsConditionalAverageEffects(randomParameters)
                
        ''' Marginal Effects of Treatment. '''
        
        _ = self._addResultsMarginalEffects(randomParameters)
        
        ''' Store update parameter objects.'''
        self.attr['parasObj'] = parasObj
        
        # Cleanup.
        self.attr.pop('grmObj', None)

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
                
        withMarginalEffects = requestObj.getAttr('withMarginalEffects')
        withAsymptotics     = requestObj.getAttr('withAsymptotics') 
        alpha               = requestObj.getAttr('alpha')

        surpEstimation      = parasObj.getAttr('surpEstimation')
        
        effectObj = clsEffects.effectCls()
        
        effectObj.lock()
        
        # Auxiliary objects.
        parasCopy = copy.deepcopy(parasObj)
        
        # Check applicability.
        if(withMarginalEffects is False): return None
        
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
                effectObj.getEffects(modelObj, parasObj, 'marginal', args)
        
        # Confidence bounds.
        if(not withAsymptotics): return None
        
        rslt = {}
        
        args = {}
        
        for parameter in parameterList:
            
            args['which'] = parameter
            
            # Simulation.
            rslt[parameter] = []
            
            for randomPara in randomParameters:
   
                parasCopy.updateValues(randomPara, isExternal = True, isAll = False)
   
                rslt[parameter].append(effectObj.getEffects(modelObj, parasCopy, 'marginal', args))
            
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

    def _addResultsAverageEffects(self, randomParameters):
        ''' Add results on average effects of treatment.
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
        
        effectObj = clsEffects.effectCls()
        
        effectObj.lock()

        withAverageEffects  = requestObj.getAttr('withAverageEffects')
        withAsymptotics     = requestObj.getAttr('withAsymptotics')
        alpha               = requestObj.getAttr('alpha')
        numDraws            = requestObj.getAttr('numDraws')
        numSims             = requestObj.getAttr('numSims')

        args = {}
        
        args['isConditional'] = False

        args['numSims']       = numSims
                
        surpEstimation      = parasObj.getAttr('surpEstimation')
                
        # Auxiliary objects.
        parasCopy = copy.deepcopy(parasObj)
        
        # Check applicability.
        if(withAverageEffects is False): return None
        
        # Simulate average effects of treatment. 
        subgroupList  = ['average', 'treated', 'untreated']
        parameterList = ['bteExPost']
    
        if(surpEstimation): parameterList += ['bteExAnte', 'cte', 'ste']
            
        for parameter in parameterList:
                
            self.attr[parameter] = {}
                
            for subgroup in subgroupList:
                    
                self.attr[parameter][subgroup] = {}
                    
                self.attr[parameter][subgroup]['estimate'] = None
                self.attr[parameter][subgroup]['pvalue']   = None
                    
                self.attr[parameter][subgroup]['confi']    = {}

        # Point estimates.
        te = effectObj.getEffects(modelObj, parasObj, 'average', args)
            
        for parameter in parameterList:
                
            for subgroup in subgroupList:
                    
                self.attr[parameter][subgroup]['estimate'] = te[parameter][subgroup]

        # Confidence bounds.
        if(not withAsymptotics): return None
        
        rslt = []
                
        for randomPara in randomParameters:
       
            parasCopy.updateValues(randomPara, isExternal = True, isAll = False)
                
            rslt.append(effectObj.getEffects(modelObj, parasCopy, 'average', args))
            
        for parameter in parameterList:
                    
            for subgroup in subgroupList:
                    
                estimate = self.attr[parameter][subgroup]['estimate']
                    
                # Collect simulations.
                teList = []
                    
                for i in range(numDraws):
                        
                    teList.append(rslt[i][parameter][subgroup])
                    
                # Process confidence interval.s
                lower, upper = scipy.stats.mstats.mquantiles(teList, \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)], axis = 0)
                    
                self.attr[parameter][subgroup]['confi']['upper'] = upper
                self.attr[parameter][subgroup]['confi']['lower'] = lower            
                    
                # Construct pvalue.
                pvalue = sum(np.sign(teList) != np.sign(estimate))/float(numDraws)
                                
                self.attr[parameter][subgroup]['pvalue'] = pvalue
        
        # Finishing.
        return None
        
    def _addResultsConditionalAverageEffects(self, randomParameters):
        ''' Add results on average effects of treatment.
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
        
        effectObj = clsEffects.effectCls()
        
        effectObj.lock()
        
        withConditionalEffects  = requestObj.getAttr('withConditionalEffects')
        withAsymptotics         = requestObj.getAttr('withAsymptotics')
        alpha                   = requestObj.getAttr('alpha')
        numDraws                = requestObj.getAttr('numDraws')
        numSims                 = requestObj.getAttr('numSims')

        args = {}
        
        args['isConditional'] = True

        args['numSims']       = numSims
             
        surpEstimation      = parasObj.getAttr('surpEstimation')
        
        # Auxiliary objects.
        parasCopy = copy.deepcopy(parasObj)
        
        # Check applicability.
        if(withConditionalEffects is False): return None
        
        # Simulate average effects of treatment. 
        subgroupList  = ['average', 'treated', 'untreated']
        parameterList = ['bteExPostCond']
    
        if(surpEstimation): parameterList += ['bteExAnteCond', 'cteCond', 'steCond']
            
        for parameter in parameterList:
                
            self.attr[parameter] = {}
                
            for subgroup in subgroupList:
                    
                self.attr[parameter][subgroup] = {}
                    
                self.attr[parameter][subgroup]['estimate'] = None
                self.attr[parameter][subgroup]['pvalue']   = None
                    
                self.attr[parameter][subgroup]['confi']    = {}

        # Point estimates.
        te = effectObj.getEffects(modelObj, parasObj, 'average', args)
            
        for parameter in parameterList:
                
            for subgroup in subgroupList:
                
                parameter = parameter.replace('Cond', '')
                
                self.attr[parameter + 'Cond'][subgroup]['estimate'] = te[parameter][subgroup]

        # Confidence bounds.
        if(not withAsymptotics): return None
        
        rslt = []
                
        for randomPara in randomParameters:
       
            parasCopy.updateValues(randomPara, isExternal = True, isAll = False)
                
            rslt.append(effectObj.getEffects(modelObj, parasObj, 'average', args))
            
        for parameter in parameterList:
            
            parameter = parameter.replace('Cond', '')
            
            for subgroup in subgroupList:
                    
                estimate = self.attr[parameter][subgroup]['estimate']
                    
                # Collect simulations.
                teList = []
                    
                for i in range(numDraws):
                        
                    teList.append(rslt[i][parameter][subgroup])
                    
                # Process confidence interval.s
                lower, upper = scipy.stats.mstats.mquantiles(teList, \
                                prob = [(alpha*0.5), (1.0 - alpha*0.5)], axis = 0)
                    
                self.attr[parameter + 'Cond'][subgroup]['confi']['upper'] = upper
                self.attr[parameter + 'Cond'][subgroup]['confi']['lower'] = lower            
                    
                # Construct pvalue.
                pvalue = sum(np.sign(teList) != np.sign(estimate))/float(numDraws)
                                
                self.attr[parameter + 'Cond'][subgroup]['pvalue'] = pvalue
        
        # Finishing.
        return None

    ''' Private methods.
    '''
    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        # Finishing.
        return True