''' Module contains all functions required to generate a dictionary
    containing the information in the initialization file.
'''

# standard library.
import numpy as np

import shlex
import json
import os

# project library
import _checkInput

''' Main function.
'''
def processInput(initFile):
    ''' Create dictionary from information in initialization file.
    '''
    # Antibugging.
    assert (os.path.exists(initFile))
    
    initDict = _constructDictionary()
    
    with open(initFile, 'r') as initFile:
        
        for line in initFile:
        
            currentLine = shlex.split(line)
            
            ''' Preprocessing.
            '''
            isEmpty, isKeyword = _processCases(currentLine)
            
            if(isEmpty):    continue
           
            if(isKeyword):  keyword = currentLine[0]
            
            if(isKeyword and (keyword in ['COST', 'BENE'])): 
                
                which = None
                group = None
                    
                which = currentLine[0]
            
                if(which == 'BENE'): group = currentLine[1]
            
            if(isKeyword): continue
            
            ''' Process major blocks.
            '''
            if(keyword ==  'DATA'):
                
                initDict = _processDATA(initDict, currentLine)
                
            if(keyword in ['BENE', 'COST']):
    
                initDict = _processEQN(initDict, currentLine, which, group)
            
            if(keyword ==  'DIST'):
                
                initDict = _processDIST(initDict, currentLine)
    
            if(keyword == 'ESTIMATION'):
                
                initDict = _processESTIMATION(initDict, currentLine)
             
            if(keyword == 'SIMULATION'):
                
                initDict = _processSIMULATION(initDict, currentLine)
    
    # Add derived information.
    initDict = _addDERIV(initDict)
    
    # Check quality.
    assert (_checkInput.checkInput(initDict) == True)
    
    # Store.
    json.dump(initDict, open('init.grm.json', 'w'), sort_keys=True, indent=2)

    # Type transformation.
    initDict = _typeTransformations(initDict)

    # Finishing.
    return initDict

''' Private auxiliary functions.
'''
def _addDERIV(initDict):
    ''' Add useful derived information for easy access.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Additional information.
    initDict['DERIV'] = {}
    
    ''' All and maximum position of covariates.
    '''
    # Initialization.
    initDict['DERIV']['pos'] = {}
    
    initDict['DERIV']['pos']['max'] = None

    initDict['DERIV']['pos']['all'] = None
    
    # Distribute source information.
    Ypos = initDict['DATA']['outcome']
    
    Dpos = initDict['DATA']['treatment']
    
    
    Bpos = initDict['BENE']['TREATED']['coeffs']['pos']
    
    Gpos = initDict['COST']['coeffs']['pos']
    
    # Construct derived information
    all_ = list(set([Ypos] + [Dpos] + Bpos + Gpos))
    
    max_ = int(max(all_))
        
    # Collect.
    initDict['DERIV']['pos']['max'] = max_

    initDict['DERIV']['pos']['all'] = all_
    
    ''' Position and number of exclusions and common elements.
    '''
    # Initialization.
    initDict['DERIV']['common'] = {}
    
    initDict['DERIV']['common']['pos'] = []
    
    initDict['DERIV']['common']['num'] = None
    
    
    initDict['DERIV']['exclBene'] = {}
    
    initDict['DERIV']['exclBene']['exAnte'] = {}
    
    initDict['DERIV']['exclBene']['exPost'] = {}
    
    
    initDict['DERIV']['exclBene']['exPost']['pos'] = []
    
    initDict['DERIV']['exclBene']['exPost']['num'] = None
    
    
    initDict['DERIV']['exclBene']['exAnte']['pos'] = []
    
    initDict['DERIV']['exclBene']['exAnte']['num'] = None
    
    
    initDict['DERIV']['exclCost'] = {}
    
    initDict['DERIV']['exclCost']['pos'] = {}
    
    initDict['DERIV']['exclCost']['num'] = None
    
    # Distribute source information.
    bene = {}
    
    bene['pos']  = np.array(initDict['BENE']['TREATED']['coeffs']['pos'])
    
    bene['info'] = np.array(initDict['BENE']['TREATED']['coeffs']['info'])
    
    
    cost = {}
    
    cost['pos']  = np.array(initDict['COST']['coeffs']['pos'])
    
    # Construct auxiliary objects.
    noCovariates = (len(bene['info']) == 0)
    
    # Construct derived information   
    initDict['DERIV']['common']['pos'] = \
        list(set(bene['pos']).intersection(cost['pos']))
    
    if(noCovariates):
        
        initDict['DERIV']['exclBene']['exAnte']['pos'] = []
                
    else:
        
        initDict['DERIV']['exclBene']['exAnte']['pos'] = \
            list(set(bene['pos'][bene['info']]).difference(cost['pos']))
    
    initDict['DERIV']['exclBene']['exPost']['pos'] = \
        list(set(bene['pos']).difference(cost['pos']))
    
    initDict['DERIV']['exclCost']['pos'] = \
        list(set(cost['pos']).difference(bene['pos']))
    
    
    initDict['DERIV']['common']['num'] = \
        len(initDict['DERIV']['common']['pos'])
        
    initDict['DERIV']['exclBene']['exAnte']['num'] = \
        len(initDict['DERIV']['exclBene']['exAnte']['pos'])
    
    initDict['DERIV']['exclBene']['exPost']['num'] = \
        len(initDict['DERIV']['exclBene']['exPost']['pos'])
    
    initDict['DERIV']['exclCost']['num'] = \
        len(initDict['DERIV']['exclCost']['pos'])
        
    # Finishing.
    return initDict
    
def _typeTransformations(initDict):
    ''' Type transformations
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Type conversions.
    for subgroup in ['TREATED', 'UNTREATED']:

        initDict['BENE'][subgroup]['coeffs']['info'] = \
            np.array(initDict['BENE'][subgroup]['coeffs']['info'])

    # Finishing.
    return initDict
    
def _constructDictionary():
    ''' Construct dictionary from initialization file.
    '''
    
    # Initialize dictionary keys.
    initDict = {}
    
    initDict['BENE'] = {}
    initDict['COST'] = {}
    
    initDict['COST']['coeffs'] = {}
    initDict['COST']['int']    = {}
    initDict['COST']['sd']     = {}
        
    initDict['COST']['coeffs']['values'] = []
    initDict['COST']['coeffs']['pos']    = []
    initDict['COST']['coeffs']['free']  = []
    
    
    initDict['COST']['sd']['values'] = []
    initDict['COST']['sd']['free'] = []
        
    initDict['COST']['int']['values'] = []
    initDict['COST']['int']['free'] = []
            
    initDict['BENE']['TREATED']   = {}
    initDict['BENE']['UNTREATED'] = {}
    
    initDict['BENE']['TREATED']   = {}
    
    initDict['BENE']['TREATED']['coeffs'] = {}
    initDict['BENE']['TREATED']['sd']     = {}
    
    initDict['BENE']['TREATED']['coeffs']['values'] = []
    initDict['BENE']['TREATED']['coeffs']['pos']    = []
    initDict['BENE']['TREATED']['coeffs']['info']   = []
    initDict['BENE']['TREATED']['coeffs']['free']  = []
    
    initDict['BENE']['TREATED']['int']     = {}
    initDict['BENE']['TREATED']['int']['values'] = []
    initDict['BENE']['TREATED']['int']['free']  = []
        
    initDict['BENE']['TREATED']['sd']     = {}
    initDict['BENE']['TREATED']['sd']['values'] = []
    initDict['BENE']['TREATED']['sd']['free']  = []    
    
    initDict['BENE']['UNTREATED']['coeffs'] = {}
    initDict['BENE']['UNTREATED']['sd']     = {}
    
    initDict['BENE']['UNTREATED']['coeffs']['values'] = []
    initDict['BENE']['UNTREATED']['coeffs']['pos']    = []
    initDict['BENE']['UNTREATED']['coeffs']['info']    = []
    initDict['BENE']['UNTREATED']['coeffs']['free']  = []

    initDict['BENE']['UNTREATED']['int']     = {}
    initDict['BENE']['UNTREATED']['int']['values'] = []
    initDict['BENE']['UNTREATED']['int']['free']  = []

    initDict['BENE']['UNTREATED']['sd']     = {}
    initDict['BENE']['UNTREATED']['sd']['values'] = []
    initDict['BENE']['UNTREATED']['sd']['free']   = []

    initDict['DATA']         = {}
        
    initDict['DIST']         = {}

    initDict['ESTIMATION'] = {}
    
    initDict['SIMULATION']   = {}

    return initDict

def _processCases(currentLine):
    ''' Process special cases of empty list and keywords.
    '''
    def _checkEmpty(currentLine):
        ''' Check whether the list is empty.
        '''
        # Antibugging.
        assert (isinstance(currentLine, list))
        
        # Evaluate list.
        isEmpty = (len(currentLine) == 0)
        
        # Check integrity.
        assert (isinstance(isEmpty, bool))
        
        # Finishing.
        return isEmpty

    def _checkKeyword(currentLine):
        ''' Check for keyword.
        '''
        # Antibugging.
        assert (isinstance(currentLine, list))
        
        # Evaluate list.
        isKeyword = False
        
        if(len(currentLine) > 0):
            
            isKeyword = (currentLine[0].isupper())
        
        # Check integrity.
        assert (isinstance(isKeyword, bool))
        
        # Finishing.
        return isKeyword
    
    ''' Main Function.
    '''
    # Antibugging.
    assert (isinstance(currentLine, list))

    # Determine indicators.
    isEmpty   = _checkEmpty(currentLine) 

    isKeyword = _checkKeyword(currentLine)
    
    # Finishing.
    return isEmpty, isKeyword

def _processEQN(initDict, currentLine, which, group):
    ''' Process DIST block.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(currentLine, list))
    assert (isinstance(which, str))
    
    if(which == 'BENE'): assert (isinstance(group, str))
    
    # Process information.   
    type_ = currentLine[0]
    
    assert (type_ in ['coeff', 'int', 'sd'])
    
    if(type_ == 'coeff'):
            
        pos    = currentLine[1]
        isFree = (currentLine[2][0] != '!')
        value  = currentLine[2].replace('!','')
        
        if(which == 'BENE'):
            
            assert (len(currentLine) == 4)

            assert (currentLine[3].upper() in ['TRUE', 'FALSE'])

            info  = (currentLine[3].upper() == 'TRUE')
                
            initDict[which][group]['coeffs']['values'] += [float(value)]
                
            initDict[which][group]['coeffs']['pos']    += [int(pos)]            
                
            initDict[which][group]['coeffs']['info']   += [info]            
            
            initDict[which][group]['coeffs']['free']   += [isFree]   
            
        if(which == 'COST'):
            
            assert (len(currentLine) == 3)
    
            initDict[which]['coeffs']['values'] += [float(value)]
                
            initDict[which]['coeffs']['pos']    += [int(pos)]            

            initDict[which]['coeffs']['free']   += [isFree]   

    if(type_ == 'int'):
        
        assert (len(currentLine) == 2)
    
        isFree = (currentLine[1][0] != '!')
        value  = currentLine[1].replace('!','')
    
        if(which == 'BENE'):
                
            initDict[which][group]['int']['values'] += [float(value)]
            initDict[which][group]['int']['free']   += [isFree]   
        
        if(which == 'COST'):
                
            initDict[which]['int']['values'] += [float(value)]  
            initDict[which]['int']['free']   += [isFree]   
                        
    if(type_ == 'sd'):

        assert (len(currentLine) == 2)
            
        isFree = (currentLine[1][0] != '!')
        value  = currentLine[1].replace('!','')
                  
        if(which == 'BENE'):
                
            initDict[which][group]['sd']['values'] += [float(value)]
            initDict[which][group]['sd']['free']   += [isFree]
                    
        if(which == 'COST'):
                
            initDict[which]['sd']['values'] += [float(value)]     
            initDict[which]['sd']['free']   += [isFree]       
    
    # Finishing.
    return initDict

def _processDIST(initDict, currentLine):
    ''' Process DIST block.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(currentLine, list))
    assert (len(currentLine) == 2)
    
    # Process information.   
    assert (currentLine[0] in ['rho0', 'rho1'])
    
    name  = currentLine[0][:-1]
    which = currentLine[0][-1]

    isFree = (currentLine[1][0] != '!')
    value  = currentLine[1].replace('!','')

    if(name not in initDict['DIST'].keys()):
        
        initDict['DIST'][name] = {}
    
    initDict['DIST'][name][which] = {}
                  
    initDict['DIST'][name][which]['value'] = float(value)
    initDict['DIST'][name][which]['free']  = isFree

    # Finishing.
    return initDict
        
def _processESTIMATION(initDict, currentLine):
    ''' Process ESTIMATION block.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(currentLine, list))
    assert (len(currentLine) == 2)
        
    # Process information.    
    keyword = currentLine[0]
    flag    = currentLine[1]
        
    # Special treatments.
    if(keyword in ['gtol', 'epsilon']):
        
        flag = float(flag)
    
    if(keyword == 'maxiter'):
        
        if(flag.upper() == 'NONE'): 

            flag = None
        
        else:
            
            flag = int(flag)

    if(keyword in ['marginal', 'conditional', 'average', 'asymptotics']):
        
        assert (flag.upper() in ['TRUE', 'FALSE'])
        
        if(flag.upper() == 'TRUE'):
            
            flag = True
        
        else:
            
            flag = False

    # Special treatments.
    if(keyword == 'alpha'):
        
        flag = float(flag)

    if(keyword in ['simulations', 'draws']):
        
        flag = int(flag)

    # Construct dictionary.
    initDict['ESTIMATION'][keyword] = flag
    
    # Finishing.
    return initDict

def _processSIMULATION(initDict, currentLine):
    ''' Process SIMULATION block.
    ''' 
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(currentLine, list))
    assert (len(currentLine) == 2)
    
    # Process information.    
    keyword = currentLine[0]
    flag    = currentLine[1]
    
    # Special treatments.
    if(keyword in ['agents', 'seed']):
        
        flag = int(flag)

    # Construct dictionary.        
    initDict['SIMULATION'][keyword] = flag
        
    # Finishing.
    return initDict

def _processDATA(initDict, currentLine):
    ''' Process DATA block.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(currentLine, list))
    assert (len(currentLine) == 2)
    
    # Process information.    
    keyword = currentLine[0]
    flag    = currentLine[1]
    
    # Special treatments.
    if(keyword in ['outcome', 'treatment']):
        
        flag = int(flag)

    if(keyword == 'agents'):
        
        if(flag.upper() == 'NONE'):
            
            flag = None
            
        else:
            
            flag = int(flag)

    # Construct dictionary.        
    initDict['DATA'][keyword] = flag
        
    # Finishing.
    return initDict
    
    