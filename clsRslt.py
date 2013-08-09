''' Module that contains the result class for the grmEstimatorToolbox.
'''

# standard library
import scipy
import copy
import random

import cPickle  as pkl
import numpy    as np

# project library

class results(object):
    ''' This class contains all results provided back to the user from the 
        maximization setup.

    ''' 
    def __init__(self):
        
        # Attach attributes.
        self.attr = {}

        self.attr['numAgents']      = None
        self.attr['isSuccess']      = None
        self.attr['parasObj']       = None
        self.attr['numDraws']       = None
        self.attr['maxiter']        = None
        self.attr['isDebug']        = None
        self.attr['covMat']         = None
        self.attr['alpha']          = None
        self.attr['fun']            = None
        self.attr['xopt']           = None
        self.attr['grad']           = None
        self.attr['hessian']        = None
        self.attr['message']        = None


        self.attr['P']              = None                                
        self.attr['D']              = None                                
        self.attr['cEval']          = None
        self.attr['zEval']          = None
        self.attr['xExPostEval']    = None
        self.attr['xExAnteEval']    = None
        
        self.attr['withConditionalEffects']  = None
        self.attr['withAverageEffects']      = None
        self.attr['withMarginalEffects']     = None
                        
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
                                                
        # Derived attributes.
        self.attr['surpEstimation'] = None
        self.attr['commonSupport']  = None
           
        # Status indicator
        self.isLocked = False
    
    ''' Public get/set methods.
    '''
    def setAttr(self, key, arg):
        ''' Set attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        assert (self._checkKey(key) == True)
        
        # Set attribute.
        self.attr[key] = arg
        
    def getAttr(self, key):
        ''' Get attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkKey(key) == True)
        
        # Select from dictionary.
        return self.attr[key]

    def unlock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True) 
        
        # Update status.
        self.isLocked = False       
                
    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == False)

        # Update status. 
        self.isLocked = True
        
        # Finalize construction.
        self._finalizeConstruction()
    
        # Quality check.
        assert (self.getStatus() == True)
        assert (self._checkIntegrity())
    
    def getStatus(self):
        ''' Get status of class instance.
        '''
        
        return self.isLocked
    
    def store(self):
        ''' Store class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True)       
        
        # Finishing.
        pkl.dump(self, open('rslt.pkl', 'wb'))
    
    ''' Calculate derived attributes.
    '''
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
        withMarginalEffects = self.getAttr('withMarginalEffects')
        surpEstimation      = self.getAttr('surpEstimation')
        alpha               = self.getAttr('alpha')
        parasObj            = self.getAttr('parasObj')    
        
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
        for parameter in parameterList:
            
            self.attr[parameter]['estimate'] = parasObj.getTreatmentParameter(parameter)
        
        # Confidence bounds.
        rslt = {}
        
        for parameter in parameterList:
            
            # Simulation.
            rslt[parameter] = []
            
            for randomPara in randomParameters:
   
                parasCopy.updateValues(randomPara)
   
                rslt[parameter].append(parasCopy.getTreatmentParameter(parameter))
            
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
        withConditionalEffects  = self.getAttr('withConditionalEffects')
        surpEstimation          = self.getAttr('surpEstimation')
        alpha                   = self.getAttr('alpha')
        numDraws                = self.getAttr('numDraws')
        parasObj                = self.getAttr('parasObj')    
        
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
        te = parasObj.getAverageEffects(isConditional = True)
            
        for parameter in parameterList:
                
            for subgroup in subgroupList:
                
                parameter = parameter.replace('Cond', '')
                
                self.attr[parameter + 'Cond'][subgroup]['estimate'] = te[parameter][subgroup]

        # Simulation.
        rslt = []
                
        for randomPara in randomParameters:
       
            parasCopy.updateValues(randomPara)
                
            rslt.append(parasCopy.getAverageEffects(isConditional = True))
            
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
        withAverageEffects  = self.getAttr('withAverageEffects')
        surpEstimation      = self.getAttr('surpEstimation')
        alpha               = self.getAttr('alpha')
        numDraws            = self.getAttr('numDraws')
        parasObj            = self.getAttr('parasObj')    
        
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
        te = parasObj.getAverageEffects()
            
        for parameter in parameterList:
                
            for subgroup in subgroupList:
                    
                self.attr[parameter][subgroup]['estimate'] = te[parameter][subgroup]

        # Simulation.
        rslt = []
                
        for randomPara in randomParameters:
       
            parasCopy.updateValues(randomPara)
                
            rslt.append(parasCopy.getAverageEffects())
            
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
        
    def _finalizeConstruction(self):
        ''' Construct derived objects.
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Distribute class attributes.
        alpha           = self.getAttr('alpha')
        numDraws        = self.getAttr('numDraws')
        parasObj        = self.getAttr('parasObj')    
        numAgents       = self.getAttr('numAgents')
        covMat          = self.getAttr('covMat')

        # Auxiliary objects.   
        paraObjs  = parasObj.getAttr('paraObjs')
        parasCopy = copy.deepcopy(parasObj)
        scale     = 1.0/numAgents
        cov       = scale*covMat
        
        # Sampling.
        np.random.seed(123)
        random.seed(456)
        
        externalValues   = parasObj.getValues(isExternal = True)
        randomParameters = np.random.multivariate_normal(externalValues, cov, numDraws)
        
        ''' Core Structural Parameters.
        '''
        counter = 0
        
        for paraObj in paraObjs:
            
            if(paraObj.getAttr('isFree') == False):
                
                paraObj.setAttr('confi', ('---', '---'))

                paraObj.setAttr('pvalue', '---')
                                
            else:
                
                rslt = []
                
                for randomPara in randomParameters:
   
                    parasCopy.updateValues(randomPara)
       
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
        
        ''' Common Support.
        '''
        self.attr['commonSupport'] = self.attr['parasObj'].getAttr('commonSupport')
        
        ''' Unconditional Average Effects of Treatment. '''
        
        _ = self._addResultsAverageEffects(randomParameters)

        ''' Conditional Average Effects of Treatment. '''
        
        _ = self._addResultsConditionalAverageEffects(randomParameters)
                
        ''' Marginal Effects of Treatment. '''
        
        _ = self._addResultsMarginalEffects(randomParameters)

    ''' Private methods.
    '''
    def _checkKey(self, key):
        ''' Check that key is present in the class attributes.
        '''
        # Antibugging.
        assert (key in self.attr.keys())
        
        # Finishing
        return True    
   
    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        
        # numAgents.
        assert (isinstance(self.attr['numAgents'], int))
        assert (self.attr['numAgents'] > 0)

        # message.
        assert (self.attr['message'] in [0, 1, 2])
        
        # hessian.
        assert (self.attr['hessian'] in ['bfgs', 'numdiff'])

        # withAverageEffects.
        assert (self.attr['withAverageEffects'] in [True, False])

        # withConditionalEffects.
        assert (self.attr['withConditionalEffects'] in [True, False])
        
        # withMarginalEffects.
        assert (self.attr['withMarginalEffects'] in [True, False])
                
        # isSuccess.
        assert (self.attr['isSuccess'] in [True, False])

        # isDebug.
        assert (self.attr['isDebug'] in [True, False])

        # gradient.
        assert (isinstance(self.attr['grad'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['grad'])))
        assert (self.attr['grad'].shape == self.attr['xopt'].shape)
        assert (self.attr['grad'].dtype == 'float')

        # maxiter.
        if(self.attr['maxiter'] is not None):
            
            assert (isinstance(self.attr['maxiter'], int))
            assert (self.attr['maxiter'] > 0)

        # Indicator.
        assert (isinstance(self.attr['D'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['D'])))
        assert (self.attr['D'].dtype == 'float')
        assert (self.attr['D'].ndim == 1)   

        # alpha.
        assert (isinstance(self.attr['alpha'], float))
        assert (0.0 < self.attr['alpha'] < 1.0)
    
        # Propensity Score.
        assert (isinstance(self.attr['P'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['P'])))
        assert (self.attr['P'].shape == (self.attr['numAgents'],))
        assert (np.all(self.attr['P'] <= 1.0) and (np.all(self.attr['P'] >= 0.0)))
                
        # zEval.
        assert (isinstance(self.attr['zEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['zEval'])))
        assert (self.attr['zEval'].dtype == 'float')
        assert (self.attr['zEval'].ndim == 1)   

        # cEval.
        assert (isinstance(self.attr['cEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['cEval'])))
        assert (self.attr['cEval'].dtype == 'float')
        assert (self.attr['cEval'].ndim == 1)   

        # suprEstimation
        assert (self.attr['surpEstimation'] in [True, False])   
    
        # Finishing.
        return True
    
