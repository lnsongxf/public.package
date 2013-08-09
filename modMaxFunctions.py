''' This module contains the criterion function of the grmEstimatorToolbox.
'''
# standard library
import  numpy           as      np

from    scipy.stats     import  norm

# project library
import clsUser

def criterionFunction(userRequest):
    ''' Negative log-likelihood function of the grmEstimatorToolbox.
    '''    
    # Antibugging.
    assert (isinstance(userRequest, clsUser.userRequest))
    assert (userRequest.getStatus() == True)
    
    # Distribute class attributes.
    numAgents = userRequest.getAttr('numAgents')
    parasObj  = userRequest.getAttr('parasObj')
    xExPost   = userRequest.getAttr('xExPost')    
    
    Y   = userRequest.getAttr('Y')
    D   = userRequest.getAttr('D')
    Z   = userRequest.getAttr('Z')

    # Distribute current parametrization.
    outcTreated   = parasObj.getParameters('outc', 'treated')
    outcUntreated = parasObj.getParameters('outc', 'untreated') 
    coeffsChoc    = parasObj.getParameters('choice', None)
    
    sdU1    = parasObj.getParameters('sd',  'U1') 
    sdU0    = parasObj.getParameters('sd',  'U0') 
    sdV     = parasObj.getParameters('sd',  'V')  
    varV    = parasObj.getParameters('var', 'V') 
        
    rhoU1V  = parasObj.getParameters('rho', 'U1,V')  
    rhoU0V  = parasObj.getParameters('rho', 'U0,V')  
    
    # Likelihood calculation.
    choiceIndices = np.dot(coeffsChoc, Z.T) 

    argOne = D*(Y - np.dot(outcTreated, xExPost.T))/sdU1 + \
            (1 - D)*(Y - np.dot(outcUntreated, xExPost.T))/sdU0

    argTwo = D*(choiceIndices - sdV*rhoU1V*argOne)/np.sqrt((1.0 - rhoU1V**2)*varV) + \
            (1 - D)*(choiceIndices - sdV*rhoU0V*argOne)/np.sqrt((1.0 - rhoU0V**2)*varV)
    
    cdfEvals = norm.cdf(argTwo)
    pdfEvals = norm.pdf(argOne)

    likl = D*(1.0/sdU1)*pdfEvals*cdfEvals + \
                (1 - D)*(1.0/sdU0)*pdfEvals*(1.0  - cdfEvals)

    # Transformations.
    likl = np.clip(likl, 1e-20, 1.0)
    
    likl = -np.log(likl)
    
    likl = likl.sum()
    
    likl = (1.0/numAgents)*likl

    # Logging.
    logFile = open('currentEval.log', 'w')

    logFile.write(' Current Evaluation' + '\n\n' + 
        '  Standard Deviation: ' + '{0:5.2f}'.format(sdV.tolist()) + '\n' + 
        '  Likelihood:         ' + '{0:5.2f}'.format(likl.tolist()))
    
    logFile.close()

    # Quality checks.
    assert (isinstance(likl, float))    
    assert (np.isfinite(likl))
    assert (likl > 0.0)

    #Finishing.        
    return likl    