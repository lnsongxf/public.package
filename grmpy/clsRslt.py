''' Module that contains the result class for the grmEstimatorToolbox.
'''

# standard library
import numpy    as np
import cPickle as pkl

import scipy
import copy
import random

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsEffects import effectCls

class results(metaCls):
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
        
        ''' Unconditional Average Effects of Treatment. '''
        
        _ = self._addResultsAverageEffects(randomParameters)

        ''' Conditional Average Effects of Treatment. '''
        
        _ = self._addResultsConditionalAverageEffects(randomParameters)
                
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

        withMarginalEffects     = requestObj.getAttr('withMarginalEffects')
        withAverageEffects      = requestObj.getAttr('withAverageEffects')
        withConditionalEffects  = requestObj.getAttr('withConditionalEffects')
        
        withAsymptotics         = requestObj.getAttr('withAsymptotics')
        surpEstimation          = parasObj.getAttr('surpEstimation')
        
        # Write results.
        isRelevant = (withMarginalEffects or withAverageEffects or withConditionalEffects)
        
        if(isRelevant):

            with open('rslt.grm.log', 'w') as file_:
                
                ''' Marginal Effects
                '''
                if(withMarginalEffects):    self._writeMarginal(file_, withAsymptotics, surpEstimation)
    
                ''' Average Effects
                '''
                if(withAverageEffects):     self._writeAverage(file_, withAsymptotics, surpEstimation)
    
                ''' Conditional Effects
                '''
                if(withConditionalEffects): self._writeConditional(file_, withAsymptotics, surpEstimation)
    
    def _writeConditional(self, file_, withAsymptotics, surpEstimation):
        ''' Write results on conditional effects of treatment.
        '''
        # Antibugging.
        assert (surpEstimation in [True, False])
        assert (withAsymptotics in [True, False])
        assert (isinstance(file_, file))
        
        # Preparation
        struct = ''' {0[0]}    {0[1]}          {0[2]} / {0[3]}    {0[4]}\n'''
    
        parameterList = ['bteExPostCond']
    
        if(surpEstimation): parameterList += ['bteExAnteCond', 'cteCond', 'steCond']
    
        # Output.            
        file_.write('\n' + ' --------------------------------------- ' + '\n' + \
                           '  Conditional Effects of Treatment       ' + '\n' + \
                           ' --------------------------------------- ' + '\n')

        for parameter in parameterList:
            
            if(parameter == 'bteExPostCond'): 
                
                title = ' Conditional Benefits of Treatment (ex post) '
                            
            if(parameter == 'bteExAnteCond'): 
                
                title = ' Conditional Benefits of Treatment (ex ante) '

            if(parameter == 'cteCond'): 
                
                title = ' Conditional Cost of Treatment '
            
            if(parameter == 'steCond'): 
                
                title = ' Conditional Surplus of Treatment  '
                            
            file_.write('\n' + title + '\n')
    
            file_.write('\n' + ' Group     Estimate    Confidence Interval  p-value' + '\n\n')
            
            for subgroup in ['average', 'treated', 'untreated']:
            
                rslt = self.attr[parameter][subgroup]
            
                est    = '{0:5.2f}'.format(rslt['estimate'])

                upper  = '---'
                lower  = '---'

                pvalue = '---'
  
                if(subgroup == 'average'):
                    
                    label = 'Average  '
                    
                if(subgroup == 'treated'):
                    
                    label = 'Treated  '                

                if(subgroup == 'untreated'):
                    
                    label = 'Untreated'
                    
                if(withAsymptotics):
                    
                    upper  = '{0:5.2f}'.format(rslt['confi']['upper'])
                    lower  = '{0:5.2f}'.format(rslt['confi']['lower'])

                    pvalue = '{0:5.2f}'.format(rslt['pvalue'])
                
                file_.write(struct.format([label, est, lower, upper, pvalue]))
                            
    def _writeAverage(self, file_, withAsymptotics, surpEstimation):
        ''' Write results on average effects of treatment.
        '''        
        # Antibugging.
        assert (surpEstimation in [True, False])
        assert (withAsymptotics in [True, False])
        assert (isinstance(file_, file))
    
        # Preparation
        struct = ''' {0[0]}    {0[1]}          {0[2]} / {0[3]}    {0[4]}\n'''
        
        parameterList = ['bteExPost']
    
        if(surpEstimation): parameterList += ['bteExAnte', 'cte', 'ste']

        # Output.            
        file_.write('\n' + ' --------------------------------------- ' + '\n' + \
                           '  Average Effects of Treatment           ' + '\n' + \
                           ' --------------------------------------- ' + '\n')

        for parameter in parameterList:
            
            if(parameter == 'bteExPost'): 
                
                title = ' Average Benefits of Treatment (ex post) '

            if(parameter == 'bteExAnte'): 
                
                title = ' Average Benefits of Treatment (ex ante) '
                
            if(parameter == 'cte'): 
                
                title = ' Average Cost of Treatment '
            
            if(parameter == 'ste'): 
                
                title = ' Average Surplus of Treatment  '
                            
            file_.write('\n' + title + '\n')
    
            file_.write('\n' + ' Group     Estimate    Confidence Interval  p-value' + '\n\n')
            
            
            for subgroup in ['average', 'treated', 'untreated']:
            
                rslt = self.attr[parameter][subgroup]
            
                est    = '{0:5.2f}'.format(rslt['estimate'])

                upper  = '---'
                lower  = '---'

                pvalue = '---'
                
                if(subgroup == 'average'):
                    
                    label = 'Average  '
                    
                if(subgroup == 'treated'):
                    
                    label = 'Treated  '                

                if(subgroup == 'untreated'):
                    
                    label = 'Untreated'
                    
                if(withAsymptotics):
                    
                    upper = '{0:5.2f}'.format(rslt['confi']['upper'])
                    lower = '{0:5.2f}'.format(rslt['confi']['lower'])
                    
                    pvalue = '{0:5.2f}'.format(rslt['pvalue'])

                file_.write(struct.format([label, est, lower, upper, pvalue]))
                        
    def _writeMarginal(self, file_, withAsymptotics, surpEstimation):
        ''' Write results on marginal effects of treatment.
        '''
        # Antibugging.
        assert (surpEstimation in [True, False])
        assert (withAsymptotics in [True, False])
        assert (isinstance(file_, file))
    
        # Preparation
        struct = '''   {0[0]}        {0[1]}          {0[2]} / {0[3]}\n'''
        idx    = np.arange(0.01, 1.00, 0.01)
    
        parameterList = ['bmteExPost']
    
        if(surpEstimation): parameterList += ['cmteExAnte', 'smteExAnte']
        
        # Output.            
        file_.write('\n' + ' --------------------------------------- ' + '\n' + \
                           '  Marginal Effects of Treatment          ' + '\n' + \
                           ' --------------------------------------- ' + '\n')

        for parameter in parameterList:
            
            points = self.attr[parameter]['estimate']
            
            if(withAsymptotics):
                
                upperBound = self.attr[parameter]['confi']['upper']
    
                lowerBound = self.attr[parameter]['confi']['lower']
        
            if(parameter == 'bmteExPost'): 
                
                title = ' Marginal Benefit of Treatment (ex post) '
            
            if(parameter == 'cmteExAnte'): 
                
                title = ' Marginal Cost of Treatment '

            if(parameter == 'smteExAnte'): 
                
                title = ' Marginal Surplus of Treatment '
                
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
                
        withMarginalEffects = requestObj.getAttr('withMarginalEffects')
        withAsymptotics     = requestObj.getAttr('withAsymptotics') 
        alpha               = requestObj.getAttr('alpha')

        surpEstimation      = parasObj.getAttr('surpEstimation')
        
        effectObj = effectCls()
        
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
   
                parasCopy.update(randomPara, version = 'external', which = 'free')
   
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
        
        effectObj = effectCls()
        
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
       
            parasCopy.update(randomPara, version = 'external', which = 'free')
                
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
        
        effectObj = effectCls()
        
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
       
            parasCopy.update(randomPara, version = 'external', which = 'free')
                
            rslt.append(effectObj.getEffects(modelObj, parasCopy, 'average', args))
            
        for parameter in parameterList:
            
            parameter = parameter.replace('Cond', '')
            
            for subgroup in subgroupList:
                    
                estimate = self.attr[parameter + 'Cond'][subgroup]['estimate']
                    
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