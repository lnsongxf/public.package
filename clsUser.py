''' Module contains the class instance that manages all things related to the
    formulation of the user's request.
'''

# standard library
import sys

import numpy            as np
import statsmodels.api  as sm

# project library
import clsParas

class userRequest(object):
    ''' This class collects all information related to the user's request.
    '''
    def __init__(self):
        
        # Attach attributes.
        self.attr = {}

        self.attr['data']       = None        
        
        self.attr['Bpos']       = None
        self.attr['Mpos']       = None
        self.attr['Cpos']       = None   
        self.attr['Ypos']       = None
        self.attr['Dpos']       = None
        
        self.attr['info']       = None
        
        # Optional arguments.
        self.attr['alpha']              = 0.05  
        self.attr['numDraws']           = 5000
        self.attr['numSims']            = 1000
        self.attr['hessian']            = 'bfgs'

        self.attr['sdVinfo']            = None
        self.attr['maxiter']            = None
        
        self.attr['withConditionalEffects']  = False
        self.attr['withAverageEffects']      = False
        self.attr['withMarginalEffects']     = True
        
        self.attr['normalization']      = None
        self.attr['isNormalized']       = False
        self.attr['surpEstimation']     = False
        self.attr['withoutPrediction']  = False
        self.attr['isDebug']            = False
        self.attr['epsilon']            = 1.4901161193847656e-08
        self.attr['gtol']               = 1e-05
                
        # Internal arguments.
        self.attr['B']              = None
        self.attr['M']              = None
        self.attr['C']              = None
        self.attr['Y']              = None
        self.attr['D']              = None
        self.attr['P']              = None
        
        self.attr['xExPost']        = None
        self.attr['xExAnte']        = None
        
        self.attr['Z']              = None
        self.attr['G']              = None

        self.attr['numAgents']      = None    
        self.attr['parasObj']       = None

        self.attr['xExPostEval']    = None
        self.attr['xExAnteEval']    = None
        self.attr['zEval']          = None
        self.attr['cEval']          = None
        
        self.attr['numCovarsExclBeneExAnte'] = None        
        self.attr['numCovarsExclBeneExPost'] = None
        self.attr['numCovarsExclCost']       = None
        self.attr['numCovarsOutc']           = None
        
        self.isLocked = False
    
    ''' Public set/get methods.
    '''
    def setAttr(self, key, arg):
        ''' Set attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        assert (self._checkKey(key) == True)
        
        # Type conversion.
        if(key == 'info'):
            
            if(arg is not None): arg = np.array(arg)
        
        # Set attribute.
        self.attr[key] = arg
        
    def getAttr(self, key):
        ''' Get attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkKey(key) == True)
        
        # Get attribute.
        return self.attr[key]
    
    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == False)

        # Update status. 
        self.isLocked = True
        
        # Construct internal objects.
        self._finalizeConstruction()
        
        # Check integrity.
        assert (self._checkIntegrity() == True)
        assert (self.getStatus() == True)
    
    def getStatus(self):
        ''' Get status of class instance.
        '''
        
        return self.isLocked
    
    def _checkKey(self, key):
        ''' Check that key is present in the class attributes.
        '''
        
        assert (key in self.attr.keys())
        
        return True
        
    def _finalizeConstruction(self):
        ''' Construction
        '''
        # Distribute class attributes.
        Ypos = self.getAttr('Ypos')
        Dpos = self.getAttr('Dpos')

        Bpos = self.getAttr('Bpos')
        Mpos = self.getAttr('Mpos')
        Cpos = self.getAttr('Cpos')
        
        info = self.getAttr('info')      
        
        # Data matrices.
        self.attr['Y'] = self.attr['data'][:, Ypos]
        self.attr['D'] = self.attr['data'][:, Dpos]

        # Covariates.      
        self.attr['B'] = None
          
        if(Bpos is not None):
        
            self.attr['B'] = self.attr['data'][:, Bpos]
        
        self.attr['M'] = self.attr['data'][:, Mpos] 
                
        self.attr['C'] = self.attr['data'][:, Cpos]
        
        # Number of agents.
        self.attr['numAgents'] = self.attr['Y'].shape[0]
        
        # Add intercept.
        self.attr['M'] = np.concatenate((self.attr['M'], \
                            np.ones((self.attr['numAgents'], 1))), axis = 1)

        # Construct outcome covariates.
        self.attr['xExPost'] = self.attr['M']
        
        self.attr['xExAnte'] = self.attr['M']
        
        if(Bpos is not None): 
            
            self.attr['xExPost'] = np.concatenate((self.attr['B'], \
                                        self.attr['xExPost']), axis = 1)
        
        if(Bpos is not None): 
            
            self.attr['xExAnte'] = np.concatenate((self.attr['B'][:, info], \
                                        self.attr['xExAnte']), axis = 1)

        # Construct cost covariates
        self.attr['G'] = np.concatenate((self.attr['M'], self.attr['C']), axis = 1)

        # Construct choice covariates.
        self.attr['Z'] = np.concatenate((self.attr['M'], self.attr['C']), axis = 1)
                
        if(Bpos is not None):
            
            self.attr['Z'] = np.concatenate((self.attr['B'][:, info], self.attr['Z']), axis = 1)
        
        # Evaluation points.
        self.attr['xExPostEval'] = self.attr['xExPost'].mean(axis = 0)

        self.attr['xExAnteEval'] = self.attr['xExAnte'].mean(axis = 0)
        
        self.attr['zEval'] = self.attr['Z'].mean(axis = 0)

        self.attr['cEval'] = self.attr['G'].mean(axis = 0)
    
        # Number of covariates.
        self.attr['numCovarsExclBeneExAnte'] = 0
        self.attr['numCovarsExclBeneExPost'] = 0

        self.attr['numCovarsExclCost']       = 0
        self.attr['numCovarsOutc']           = 0
         
        if(Bpos is not None):
            
            self.attr['numCovarsExclBeneExPost'] = len(Bpos)
            self.attr['numCovarsExclBeneExAnte'] = sum(info)     
            
        self.attr['numCovarsExclCost'] = len(Cpos)
        self.attr['numCovarsOutc']     = self.attr['xExPost'].shape[1]
        
        # Process normalization.
        if(self.attr['normalization'] is not None):
            
            self.attr['isNormalized'] = True
        
        # Prediction step.
        self.attr['withoutPrediction'] = False
        
        if(Bpos is not None):
            
            self.attr['withoutPrediction'] = (sum(info) == len(Bpos))
                
        # Surplus estimated.
        self.attr['surpEstimation'] = False
        
        if(info is not None):
            
            self.attr['surpEstimation'] = ((sum(info) > 0) and  \
                    (self.attr['isNormalized'] == False))

        # Calculate Propensity Score.
        sys.stdout = open('/dev/null', 'w')   
             
        rslt = sm.Probit(self.attr['D'], self.attr['Z'])
        P    = rslt.predict(rslt.fit().params)
            
        sys.stdout = sys.__stdout__
        
        self.attr['P'] = P
        
        # Initialize container.
        parasObj = clsParas.parasCls() 
         
        # Initialize Outcome Model (Treated).
        coeffs, sd = self._getStartingValues('treated')
        
        if(Bpos is not None):

            for _ in Bpos:
                
                coeff = coeffs.pop(0)

                parasObj.addParameter('outc', 'treated', coeff, True, (None, None))
                                
        for _ in Mpos + [1]:
            
            coeff = coeffs.pop(0)
            
            parasObj.addParameter('outc', 'treated', coeff, True, (None, None))
                    
        parasObj.addParameter('sd', 'U1', sd.pop(0), True, (0.01, None))  
           
        # Antibugging.
        assert (len(coeffs) == 0)
        assert (len(sd) == 0)
       
        # Initialize Outcome Model (Untreated).
        coeffs, sd = self._getStartingValues('untreated')

        if(Bpos is not None):
        
            for _ in Bpos:
                
                coeff = coeffs.pop(0)
                                
                parasObj.addParameter('outc', 'untreated', coeff, True, (None, None)) 
                
        for _ in Mpos + [1]:
            
            coeff = coeffs.pop(0)
                            
            parasObj.addParameter('outc', 'untreated', coeff, True, (None, None))    
                    
        parasObj.addParameter('sd', 'U0', sd.pop(0), True, (0.01, None))  

        # Antibugging.
        assert (len(coeffs) == 0)
        assert (len(sd) == 0)
       
        # Initialize parameters for cost shifters.
        coeffs, sd = self._getStartingValues('cost')
        
        for _ in Mpos + [1]:
            
            coeff = coeffs.pop(0)
            
            parasObj.addParameter('cost', None, coeff, True, (None, None))  
                    
        for _ in Cpos:
            
            coeff = coeffs.pop(0)
            
            parasObj.addParameter('cost', None, coeff, True, (None, None))          
            
        # Antibugging.
        assert (len(coeffs) == 0)
        assert (len(sd) == 1)
           
        # Initialize coefficients of correlations.
        parasObj.addParameter('rho', 'U1,V', 0.00, True, (-0.99, 0.99))    
        
        parasObj.addParameter('rho', 'U0,V', 0.00, True, (-0.99, 0.99))      
        
        # Set standard deviation of V.
        if(self.attr['isNormalized']):

            value = self.attr['normalization']
            
            parasObj.addParameter('sd', 'V', value, False, (0.01, None))    

        else:

            value = 1.0
            bound = None
            
            if(self.attr['sdVinfo'] is not None): 
                
                value = self.attr['sdVinfo'][0]
                bound = self.attr['sdVinfo'][1]
            
            parasObj.addParameter('sd', 'V', value, True, (0.01, bound))    

        # Number of exclusive shifters.
        parasObj.setAttr('numCovarsExclBeneExPost', self.attr['numCovarsExclBeneExPost'])

        parasObj.setAttr('numCovarsExclBeneExAnte', self.attr['numCovarsExclBeneExAnte'])
        
        parasObj.setAttr('numCovarsExclCost', self.attr['numCovarsExclCost'])

        # Number of outcome shifters.
        parasObj.setAttr('numCovarsOutc', self.attr['numCovarsOutc'])
        
        # Covariates for prediction step.
        parasObj.setAttr('xExAnte', self.attr['xExAnte'])
        
        parasObj.setAttr('xExPost', self.attr['xExPost'])

        parasObj.setAttr('G', self.attr['G'])
        
        parasObj.setAttr('Z', self.attr['Z'])

        parasObj.setAttr('D', self.attr['D'])
                
        parasObj.setAttr('numAgents', self.attr['numAgents'])

        parasObj.setAttr('normalization', self.attr['normalization'])

        parasObj.setAttr('isNormalized', self.attr['isNormalized'])

        parasObj.setAttr('withoutPrediction', self.attr['withoutPrediction'])

        parasObj.setAttr('xExPostEval', self.attr['xExPostEval'])

        parasObj.setAttr('xExAnteEval', self.attr['xExAnteEval'])
        
        parasObj.setAttr('cEval', self.attr['cEval'])

        parasObj.setAttr('zEval', self.attr['zEval'])       

        parasObj.setAttr('numSims', self.attr['numSims']) 
                                                                                                                                 
        parasObj.lock()
        
        self.attr['parasObj'] = parasObj
    
    def _getStartingValues(self, which):
        ''' Get starting values.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (which in ['treated', 'untreated', 'cost'])
        
        # Data selection.
        Y = self.getAttr('Y')
        D = self.getAttr('D')
        
        X = self.getAttr('xExPost')
        G = self.getAttr('G')
        
        # Subset selection.
        if(which == 'treated'):
            
            Y = Y[D == 1]
            X = X[(D == 1), :]
        
        elif(which == 'untreated'):
            
            Y = Y[D == 0]
            X = X[(D == 0), :]
        
        # Model selection. 
        if(which in ['treated', 'untreated']):
            
            olsRslt = sm.OLS(Y, X).fit()

            coeffs = olsRslt.params
            sd     = np.array(np.sqrt(olsRslt.scale))
        
        elif(which == 'cost'):
            
            sys.stdout = open('/dev/null', 'w')
            
            probitRslt =  sm.Probit(D, G).fit()
            
            coeffs = -probitRslt.params
            sd     = np.array(1.0)
            
            sys.stdout = sys.__stdout__
            
        # Quality checks.
        assert (isinstance(coeffs, np.ndarray))
        assert (isinstance(sd, np.ndarray))
        
        assert (np.all(np.isfinite(coeffs)))                
        assert (np.all(np.isfinite(sd)))
    
        assert (coeffs.ndim == 1)
        assert (sd.ndim == 0)
        
        assert (coeffs.dtype == 'float')
        assert (sd.dtype == 'float')

        # Type conversion.
        coeffs = coeffs.tolist()
        sd     = [sd.tolist()]
        
        # Finishing.
        return coeffs, sd
        
    def _checkIntegrity(self):
        ''' Check integrity of user request.
        '''
        
        # Dataset.
        assert (isinstance(self.attr['data'], np.ndarray))
        assert (self.attr['data'].dtype == 'float')
        assert (self.attr['data'].ndim == 2)
        
        # Position of benefit shifters.
        if(self.attr['Bpos'] is not None):
            
            assert (isinstance(self.attr['Bpos'], list))
            assert (all(isinstance(int_, int) for int_ in self.attr['Bpos']))
        
        # Position of common elements.
        assert (isinstance(self.attr['Mpos'], list))
        assert (all(isinstance(int_, int) for int_ in self.attr['Mpos']))
        
        # Position of cost elements.
        assert (isinstance(self.attr['Cpos'], list))
        assert (all(isinstance(int_, int) for int_ in self.attr['Cpos']))
       
        # Position of outcome.
        assert (isinstance(self.attr['Ypos'], int))
        
        # Position of treatment indicator.    
        assert (isinstance(self.attr['Dpos'], int))    

        # Information set.
        if(self.attr['info'] is not None):

            assert (isinstance(self.attr['info'], np.ndarray))
            assert (self.attr['info'].dtype == 'bool')    
            assert (self.attr['info'].shape == (len(self.attr['Bpos']),))
        
        if(self.attr['Bpos'] is None):
            
            assert (self.attr['info'] is None)

        # Normalization.
        if(self.attr['normalization'] is not None):
            
            assert (isinstance(self.attr['normalization'], float))
            assert (self.attr['normalization'] > 0.0)
        
        # isNormalized.
        assert (self.attr['isNormalized'] in [True, False])
        
        if(self.attr['isNormalized']):
            
            assert (self.attr['sdVinfo'] is None)

        # Check identification.
        if((self.attr['isNormalized'] == False)  \
           and (self.attr['isDebug'] == False)):
            
            assert (self.attr['Bpos'] is not None)
            assert (self.attr['info'] is not None)
            assert (sum(self.attr['info']) > 0)

        # suprEstimation
        assert (self.attr['surpEstimation'] in [True, False])   
        
        # withoutPrediction. 
        assert (self.attr['withoutPrediction'] in [True, False])             

        # withConditionalAverageEffects. 
        assert (self.attr['withConditionalEffects'] in [True, False])   
        
        # withAverageEffects. 
        assert (self.attr['withAverageEffects'] in [True, False])       

        # withMarginalEffects. 
        assert (self.attr['withMarginalEffects'] in [True, False])  
        
        # sdVinfo.
        if(self.attr['sdVinfo'] is not None):
            
            value = self.attr['sdVinfo'][0]
            bound = self.attr['sdVinfo'][1]
                                                              
            # Starting value
            assert (isinstance(value, float))
            assert (value > 0.0)
            
            # Upper bound.         
            if(bound is not None):

                assert (isinstance(bound, float))
                assert (bound > 0.0)
                assert (value < bound)
                
        # Maximum iteration.
        if(self.attr['maxiter'] is not None):
            
            assert (isinstance(self.attr['maxiter'], int))
            assert (self.attr['maxiter'] > 0)
        
        # isDebug.
        assert (self.attr['isDebug'] in [True, False])
        
        # alpha.
        assert (isinstance(self.attr['alpha'], float))
        assert (0.0 < self.attr['alpha'] < 1.0)

        # gtol.
        assert (isinstance(self.attr['gtol'], float))
        assert (self.attr['gtol'] > 0.00)
        
        # epsilon.
        assert (isinstance(self.attr['epsilon'], float))
        assert (self.attr['epsilon'] > 0.00)
                
        # Benefit shifters. 
        if(self.attr['Bpos'] is not None):
                    
            assert (isinstance(self.attr['B'], np.ndarray))
            assert (np.all(np.isfinite(self.attr['B'])))
            assert (self.attr['B'].dtype == 'float')
            assert (self.attr['B'].ndim == 2)
            
        # Common elements.
        if(self.attr['Mpos'] is not None):
                    
            assert (isinstance(self.attr['M'], np.ndarray))
            assert (np.all(np.isfinite(self.attr['M'])))
            assert (self.attr['M'].dtype == 'float')
            assert (self.attr['M'].ndim == 2)        

        # Cost shifters.
        assert (isinstance(self.attr['C'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['C'])))
        assert (self.attr['C'].dtype == 'float')
        assert (self.attr['C'].ndim == 2)        

        # Outcome.
        assert (isinstance(self.attr['Y'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['Y'])))
        assert (self.attr['Y'].dtype == 'float')
        assert (self.attr['Y'].ndim == 1)       

        # Indicator.
        assert (isinstance(self.attr['D'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['D'])))
        assert (self.attr['D'].dtype == 'float')
        assert (self.attr['D'].ndim == 1)       
    
        # xExPost.
        assert (isinstance(self.attr['xExPost'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExPost'])))
        assert (self.attr['xExPost'].dtype == 'float')
        assert (self.attr['xExPost'].ndim == 2)      
 
        # xExPost.
        assert (isinstance(self.attr['xExAnte'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExAnte'])))
        assert (self.attr['xExAnte'].dtype == 'float')
        assert (self.attr['xExAnte'].ndim == 2)    
               
        # Z.
        assert (isinstance(self.attr['Z'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['Z'])))
        assert (self.attr['Z'].dtype == 'float')
        assert (self.attr['Z'].ndim == 2)    

        # G.
        assert (isinstance(self.attr['G'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['G'])))
        assert (self.attr['G'].dtype == 'float')
        assert (self.attr['G'].ndim == 2)    

        # numAgents.
        assert (isinstance(self.attr['numAgents'], int))
        assert (self.attr['numAgents'] > 0)
        
        # ParasObj.
        assert (isinstance(self.attr['parasObj'], clsParas.parasCls))
        assert (self.attr['parasObj'].getStatus() == True)
        
        # xExPostEval.
        assert (isinstance(self.attr['xExPostEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExPostEval'])))
        assert (self.attr['xExPostEval'].dtype == 'float')
        assert (self.attr['xExPostEval'].ndim == 1)   

        # xExAnteEval.
        assert (isinstance(self.attr['xExAnteEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExAnteEval'])))
        assert (self.attr['xExAnteEval'].dtype == 'float')
        assert (self.attr['xExAnteEval'].ndim == 1)   

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
        
        # numCovarsExclBeneExAnte.
        assert (isinstance(self.attr['numCovarsExclBeneExAnte'], int))
        assert (self.attr['numCovarsExclBeneExAnte'] >= 0)
        
        # numCovarsExclBeneExPost.
        assert (isinstance(self.attr['numCovarsExclBeneExPost'], int))
        assert (self.attr['numCovarsExclBeneExPost'] >= 0)
                
        # numCovarsExclCost.
        assert (isinstance(self.attr['numCovarsExclCost'], int))
        assert (self.attr['numCovarsExclCost'] >= 0)                

        # numCovarsOutc.
        assert (isinstance(self.attr['numCovarsOutc'], int))
        assert (self.attr['numCovarsOutc'] >= 0)                      
        
        # hessian.
        assert (self.attr['hessian'] in ['bfgs', 'numdiff'])
        
        # Propensity Score.
        assert (isinstance(self.attr['P'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['P'])))
        assert (self.attr['P'].shape == (self.attr['numAgents'],))
        assert (np.all(self.attr['P'] <= 1.0) and (np.all(self.attr['P'] >= 0.0)))

        # Finishing.
        return True
        