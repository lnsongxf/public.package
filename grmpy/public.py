""" Interface to public functions of the grmToolbox.
"""

# standard library
import numdifftools as nd
import numpy as np
import sys
import random
import shutil
import glob
import os

try:
   import cPickle as pkl
except:
   import pickle as pkl


# project library
from grmpy.tools.auxiliary import cleanup, createMatrices
from grmpy.user.init_interface import initialize
from grmpy.clsGrm import grmCls
from grmpy.clsCrit import critCls
from grmpy.user._createDictionary import processInput
from grmpy.clsMax import _scipyWrapperFunction as evaluate
from grmpy.clsMax import maxCls
from grmpy.clsGrm import grmCls
from grmpy.clsRslt import results

def clean():
    ''' Cleanup from previous estimation run.
    '''
    # Construct files list.
    fileList = glob.glob('*.grmpy.*')


    # Remove information from simulated data.
    for file_ in ['*.infos.grmpy.out', '*.paras.grmpy.out']:

        try:

            fileList.remove(glob.glob(file_)[0])

        except:

            pass

    # Cleanup
    for file_ in fileList:

        try:

            os.remove(file_)

        except OSError:

            pass

def estimate(init = 'init.ini', resume = False, useSimulation = False):
    ''' Estimate specified model.
    '''
    # Cleanup
    cleanup(resume)

    #Process initialization file.
    modelObj, parasObj, requestObj, _ = initialize(init, useSimulation)

    # Process resume.
    if(resume):

        # Antibugging.
        assert (os.path.isfile('stepParas.grmpy.out'))

        # Update parameter objects.
        parasObj = _updateParameters(parasObj)

    else:

        paras = parasObj.getValues(version = 'internal', which = 'all')

        np.savetxt('stepParas.grmpy.out', paras, fmt = '%25.12f')

    # Initialize container
    grmObj = grmCls()

    grmObj.setAttr('modelObj', modelObj)

    grmObj.setAttr('requestObj', requestObj)

    grmObj.setAttr('parasObj', parasObj)

    grmObj.lock()

    # Set random seed.
    np.random.seed(123)

    # Distribute class attributes.
    requestObj = grmObj.getAttr('requestObj')

    hessian   = requestObj.getAttr('hessian')

    withAsymptotics = requestObj.getAttr('withAsymptotics')

    # Distribute auxiliary objects.
    maxObj = maxCls(grmObj)

    maxObj.lock()

    sys.stdout = open('/dev/null', 'w')

    maxRslt = maxObj.maximize()

    sys.stdout = sys.__stdout__

    # Distribute results.
    xopt = maxRslt['xopt']

    # Approximate hessian.
    covMat  = np.tile(np.nan, (len(xopt), len(xopt)))

    if(withAsymptotics):

        if(hessian == 'bfgs'):

            covMat = maxRslt['covMat']

        elif(hessian == 'numdiff'):

            critFunc = maxObj.getAttr('critFunc')

            ndObj    = nd.Hessian(lambda x: evaluate(x, critFunc))
            hess     = ndObj(xopt)
            covMat   = np.linalg.pinv(hess)

        pkl.dump(covMat, open('covMat.grmpy.pkl', 'wb'))

    # Construct result class.
    rslt = results()

    rslt.setAttr('grmObj', grmObj)

    rslt.setAttr('maxRslt', maxRslt)

    rslt.setAttr('covMat', covMat)

    rslt.lock()

    rslt.store('rslt.grmpy.pkl')

    return rslt

def test():
    """ Run nose tester.
    """
    base = os.getcwd()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    os.chdir('tests')

    os.system('nosetests test.py')

    os.chdir(base)

def simulate(init = 'init.ini', update = False):
    ''' Simulate dataset for grmToolbox.
    '''

    isMock = _createMock(init)

    ''' Process initialization file.
    '''
    _, parasObj, _, initDict = initialize(init, isSimulation = True)

    ''' Distribute information.
    '''
    target = initDict['SIMULATION']['target']

    seed   = initDict['SIMULATION']['seed']

    np.random.seed(seed); random.seed(seed)

    ''' Update parameter class.
    '''
    if(update): parasObj = _updateParameters(parasObj)

    ''' Create simulated dataset.
    '''
    if(isMock): os.remove(initDict['DATA']['source'])

    simAgents   = initDict['SIMULATION']['agents']

    max_        = initDict['DERIV']['pos']['max']

    simDat      = np.empty((simAgents, max_ + 1), dtype = 'float')

    simDat[:,:] = np.nan


    simDat = _simulateExogenous(simDat, initDict)

    simDat = _simulateEndogenous(simDat, parasObj, initDict)

    ''' Update for prediction step.
    '''
    rslt = createMatrices(simDat, initDict)

    parasObj.unlock()

    parasObj.setAttr('xExAnte', rslt['xExAnte'])

    parasObj.setAttr('xExPost', rslt['xExPost'])

    parasObj.lock()

    ''' Save dataset.
    '''
    np.savetxt(target, simDat, fmt = '%15.10f')

    likl = _getLikelihood(init)

    _writeInfo(parasObj, target, rslt, likl)

''' Auxiliary functions.
'''
def _updateParameters(parasObj):
    ''' Update parameter object if possible.
    '''
    # Antibugging.
    assert (parasObj.getStatus() == True)

    # Update.
    hasStep = (os.path.isfile('stepParas.grmpy.out'))

    if(hasStep):

        internalValues = np.array(np.genfromtxt('stepParas.grmpy.out'), dtype = 'float', ndmin = 1)

        parasObj.update(internalValues, version = 'internal', which = 'all')

    # Finishing.
    return parasObj

def _getLikelihood(init):
    ''' Calculate likelihood for simulated dataset at true parameter values.
    '''
    # Antibugging.
    assert (isinstance(init, str))

    # Process model ingredients.
    modelObj, parasObj, requestObj, _ = initialize(init, True)

    # Initialize container.
    grmObj = grmCls()

    grmObj.setAttr('modelObj', modelObj)

    grmObj.setAttr('requestObj', requestObj)

    grmObj.setAttr('parasObj', parasObj)

    grmObj.lock()


    critObj = critCls(grmObj)

    critObj.lock()

    # Evaluate at true values.
    x    = parasObj.getValues('external', 'free')

    likl = evaluate(x, critObj)

    # Cleanup.
    for file_ in ['grmToolbox.grmpy.log', 'stepParas.grmpy.out', \
                  'startParas.grmpy.out']:

        os.remove(file_)

    # Finishing.
    return likl

def _createMock(init):
    ''' Create a mock dataset which allows for use of existing routines
        in the case of a missing source dataset.
    '''

    initDict = processInput(init)

    isMock = (os.path.exists(initDict['DATA']['source']) == False)

    obsAgents = initDict['DATA']['agents']

    pos  = initDict['DATA']['treatment']

    max_      = initDict['DERIV']['pos']['max']

    simDat      = np.empty((obsAgents, max_ + 1), dtype = 'float')

    simDat[:,:] = np.random.randn(obsAgents, max_ + 1)

    simDat[:,pos] = np.random.random_integers(0, 1, obsAgents)

    source = initDict['DATA']['source']

    np.savetxt(source, simDat, fmt = '%15.10f')

    # Finishing.
    return isMock

def _simulateEndogenous(simDat, parasObj, initDict):
    ''' Simulate the endogenous characteristics such as choices and outcomes.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(simDat, np.ndarray))
    assert (parasObj.getStatus() == True)

    # Distribute information.
    simAgents = initDict['SIMULATION']['agents']

    outcome   = initDict['DATA']['outcome']

    treatment = initDict['DATA']['treatment']

    all_      = initDict['DERIV']['pos']['all']

    # Sampling of unobservables.
    varV  = parasObj.getParameters('var',  'V')

    varU1 = parasObj.getParameters('var',  'U1')

    varU0 = parasObj.getParameters('var',  'U0')

    mean   = np.tile(0.0, 3)

    covMat = np.diag([varU1, varU0, varV])

    covMat[2,0] = covMat[0,2] = parasObj.getParameters('cov', 'U1,V')

    covMat[2,1] = covMat[1,2] = parasObj.getParameters('cov', 'U0,V')

    U1, U0, V = np.random.multivariate_normal(mean, covMat, simAgents).T

    # Create data matrices.
    rslt    = createMatrices(simDat, initDict)

    xExPost = rslt['xExPost']

    Z       = rslt['Z']

    # Simulate choices.
    coeffsChoc = parasObj.getParameters('choice', None)

    D = (np.dot(coeffsChoc, Z.T) - V > 0.0)

    # Potential Outcomes
    outcTreated   = parasObj.getParameters('outc', 'treated')
    outcUntreated = parasObj.getParameters('outc', 'untreated')

    Y1 = np.dot(outcTreated, xExPost.T)   + U1
    Y0 = np.dot(outcUntreated, xExPost.T) + U0

    Y = D*Y1 + (1 - D)*Y0

    simDat[:,outcome]   = Y
    simDat[:,treatment] = D

    # Quality checks.
    assert (isinstance(simDat, np.ndarray))
    assert (np.all(np.isfinite(simDat[:,all_])))
    assert (simDat.dtype == 'float')

    # Finishing.
    return simDat

def _simulateExogenous(simDat, initDict):
    ''' Simulate the exogenous characteristics by filling up the data frame
        with random deviates of the exogenous characteristics from the
        observed dataset.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(simDat, np.ndarray))

    # Distribute information.
    source    = initDict['DATA']['source']

    simAgents = initDict['SIMULATION']['agents']

    all_      = initDict['DERIV']['pos']['all']

    outcome   = initDict['DATA']['outcome']

    treatment = initDict['DATA']['treatment']

    # Restrict to exogenous positions.
    for pos in [outcome, treatment]:

        all_.remove(pos)

    # Simulate endogenous characteristics.
    hasSource   = (os.path.exists(source) == True)

    if(hasSource):

        obsDat    = np.genfromtxt(source)

        obsAgents = obsDat.shape[0]

        if(obsAgents == simAgents):

            idx_ = range(obsAgents)

        else:

            idx_ = np.random.randint(0, obsAgents, size = simAgents)


        for pos in all_:

            simDat[:,pos] = obsDat[idx_,pos]

    else:

        for pos in all_:

            simDat[:,pos] = np.random.randn(simAgents)

    # Quality checks.
    assert (isinstance(simDat, np.ndarray))
    assert (np.all(np.isfinite(simDat[:,all_])))
    assert (simDat.dtype == 'float')

    # Finishing.
    return simDat

def _writeInfo(parasObj, target, rslt, likl):
    ''' Write out some additional infos about the simulated dataset.
    '''

    # Auxiliary objects.
    fileName     = target.split('.')[0]

    numAgents    = str(len(rslt['Y']))

    numTreated   = np.sum(rslt['D'] == 1)

    numUntreated = np.sum(rslt['D'] == 0)

    fval         = str(likl)

    # Write out structural parameters.
    paras = parasObj.getValues(version = 'internal', which = 'all')

    np.savetxt(fileName + '.paras.grmpy.out', paras, fmt = '%15.10f')

    # Write out information on agent experiences.
    with open(fileName + '.infos.grmpy.out', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Observations: ' + numAgents + '\n')

        file_.write('   Function Value:         ' + fval + '\n\n')

        string  = '''{0[0]:<10} {0[1]:>12}\n'''

        file_.write('   Choices:  \n\n')

        file_.write(string.format(['     Treated  ', numTreated]))

        file_.write(string.format(['     Untreated', numUntreated]))

        file_.write('\n\n')


        string  = '''{0[0]:<10} {0[1]:15.5f}\n'''

        file_.write('   Outcomes:  \n\n')

        file_.write(string.format(['     Treated  ', np.mean(rslt['Y'][rslt['D'] == 1])]))

        file_.write(string.format(['     Untreated', np.mean(rslt['Y'][rslt['D'] == 0])]))


        file_.write('\n\n')