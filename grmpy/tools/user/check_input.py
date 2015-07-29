""" Module that checks the user input after processing of the initialization
    file.
"""

# standard library
import numpy as np

''' Main function.
'''
def check_input(initDict):
    """ Check the input from the initialization file.
    """
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Check keys.
    keys = set(['DATA', 'BENE', 'COST', 'RHO', 'ESTIMATION', 'SIMULATION', 'DERIV'])
    
    assert (keys == set(initDict.keys()))
    
    ''' Check subgroups.
    '''
    _check_eqn(initDict)
    
    _check_data(initDict)

    _check_estimation(initDict)
    
    _check_simulation(initDict)

    _check_rho(initDict)
    
    _check_msc(initDict)
    
    _check_deriv(initDict)

    # Finishing
    return True

''' Private auxiliary functions.
''' 
def _check_deriv(initDict):
    """ Check derived information.
    """
    assert (isinstance(initDict, dict))

    ''' Positions.
    '''
    # Distribute
    all_ = initDict['DERIV']['pos']['all']

    max_ = initDict['DERIV']['pos']['max']
    
    # Checks
    assert (isinstance(max_, int))
    assert (max_ > 0)
    assert (isinstance(all_, list))
    assert (np.all(np.isfinite(all_)))
    assert (all(isinstance(pos, int) for pos in all_))
    
    ''' Exclusions.
    '''
    # Benefit shifters
    for info in ['ex_post', 'ex_ante']:
        
        list_ = initDict['DERIV']['excl_bene'][info]['pos']
        
        num = initDict['DERIV']['excl_bene'][info]['num']
        
        assert (isinstance(list_, list))
        assert (isinstance(num, int))
        assert (all(isinstance(pos, np.int64) for pos in list_))
        assert (num >= 0)
    
    # Cost shifters
    list_ = initDict['DERIV']['excl_cost']['pos']
    num = initDict['DERIV']['excl_cost']['num']
        
    assert (isinstance(list_, list))
    assert (isinstance(num, int))
        
    assert (all(isinstance(pos, np.int64) for pos in list_))
    assert (num >= 0)

    # Common elements.
    list_ = initDict['DERIV']['common']['pos']
    
    pos = initDict['DERIV']['common']['num']
    
    assert (isinstance(list_, list))
    assert (isinstance(num, int))
        
    assert (all(isinstance(pos, np.int64) for pos in list_))
    assert (num >= 0)  
    
    # Finishing.
    return True


def _check_msc(initDict):
    """ Check selected additional constraints.
    """
    assert (isinstance(initDict, dict))
    
    ''' Check that the covariates in the treated and untreated state are
        identical.
    '''
    treated = set(initDict['BENE']['TREATED']['coeffs']['pos'])
    untreated = set(initDict['BENE']['UNTREATED']['coeffs']['pos'])
    
    assert (treated == untreated)
    
    treated = initDict['BENE']['TREATED']['coeffs']['info']
    untreated = initDict['BENE']['UNTREATED']['coeffs']['info']
    numInfo = len(treated)
    
    assert (all((treated[i] == untreated[i]) for i in range(numInfo)))
    
    ''' Check that there are no duplicates in either benefit or cost equation.
    '''
    treated = initDict['BENE']['TREATED']['coeffs']['pos']
    untreated = initDict['BENE']['UNTREATED']['coeffs']['pos']
    cost = initDict['BENE']['UNTREATED']['coeffs']['pos']
    
    for obj in [treated, untreated, cost]:
        assert (len(obj) == len(set(obj)))
    
    ''' Check identification.
    '''
    is_free = initDict['COST']['sd']['free'][0]
    
    if is_free:
        assert (initDict['DERIV']['excl_bene']['ex_ante']['num'] > 0)

    ''' Check that outcome and treated are not columns in either benefit or
        cost equations.
    '''
    treated = set(initDict['BENE']['TREATED']['coeffs']['pos'])
    untreated = set(initDict['BENE']['UNTREATED']['coeffs']['pos'])
    cost = set(initDict['BENE']['UNTREATED']['coeffs']['pos'])
    
    y = set([initDict['DATA']['outcome']])
    d = set([initDict['DATA']['treatment']])

    for obj in [treated, untreated, cost]:
        
        assert (len(obj.intersection(y)) == 0)
        assert (len(obj.intersection(d)) == 0)

    ''' Check that at least one regressor in choice, either ex ante benefit
        or cost shifters.
    '''
    bene = len(initDict['BENE']['TREATED']['coeffs']['pos'])
    cost = len(initDict['COST']['coeffs']['pos'])
    
    no_benefit_shifters = (bene == 0)
    no_cost_shifters = (cost == 0)
    
    if no_benefit_shifters:
        assert (cost > 0)
        
    if no_cost_shifters:
        info = sum(initDict['BENE']['TREATED']['coeffs']['info'])
        assert (info > 0)

    ''' Check that no covariates that are unknown to the agent at the time of
        the treatment decision are specified as cost shifters.
    '''   
    info = np.array(initDict['BENE']['TREATED']['coeffs']['info'])
    
    benefit_shifters = (len(info) > 0)
    
    if benefit_shifters:
       
        pos = np.array(initDict['BENE']['TREATED']['coeffs']['pos'])
        unknown = set(pos[info == False])
        pos = set(initDict['COST']['coeffs']['pos'])
     
        assert (len(unknown.intersection(pos)) == 0)
        
    # Finishing
    return True


def _check_eqn(initDict):
    """ Check EQN block.
    """
    assert (isinstance(initDict, dict))

    # Check keys.
    assert (set(['TREATED', 'UNTREATED']) == set(initDict['BENE'].keys()))
    
    ''' BENE.
    '''
    for group in ['TREATED', 'UNTREATED']:
        
        for subgroup in ['coeffs', 'sd', 'int']:

            positions = None            
            
            values = None
            
            infos = None
            
            free = None
            
            if subgroup == 'coeffs':
                
                positions = initDict['BENE'][group][subgroup]['pos']
                
                values = initDict['BENE'][group][subgroup]['values']
                
                infos = initDict['BENE'][group][subgroup]['info']
                
                free = initDict['BENE'][group][subgroup]['free']
                
            if subgroup in ['sd', 'int']:
                
                values = initDict['BENE'][group][subgroup]['values']
                
                free = initDict['BENE'][group][subgroup]['free']
                
            # Position
            if positions is not None:
                assert (isinstance(positions, list))             
                assert (all(isinstance(pos, int) for pos in positions))
                assert ((pos >= 0) for pos in positions)
   
            # Values
            if values is not None :
                assert (isinstance(values, list))             
                assert (all(isinstance(value, float) for value in values))
            
            # Information and Free
            for objs in [infos, free]:
                if objs is not None:
                    assert (isinstance(objs, list))             
                    assert (all(isinstance(obj, bool) for obj in objs))    

    ''' COST.
    '''
    for subgroup in ['coeffs', 'sd', 'int']:

        positions = None
        values = None
        free = None
            
        if subgroup == 'coeffs':
            positions = initDict['COST'][subgroup]['pos']
            values = initDict['COST'][subgroup]['values']
            free = initDict['COST'][subgroup]['free']
                
        if subgroup in ['sd', 'int']:
            values = initDict['COST'][subgroup]['values']
            free = initDict['COST'][subgroup]['free']
                
        # Position
        if positions is not None:
            assert (isinstance(positions, list))             
            assert (all(isinstance(pos, int) for pos in positions))
            assert ((pos >= 0) for pos in positions)
   
        # Values
        if values is not None:
            assert (isinstance(values, list))             
            assert (all(isinstance(value, float) for value in values))
            
        # Free
        if free is not None:
            assert (isinstance(free, list))             
            assert (all(isinstance(obj, bool) for obj in free))                
    
    # Finishing
    return True


def _check_data(initDict):
    """ Check DATA block.
    """
    assert (isinstance(initDict, dict))

    # Check keys.
    keys = set(['source', 'agents', 'outcome', 'treatment'])

    assert (keys == set(initDict['DATA'].keys()))

    # Distribute elements.
    source = initDict['DATA']['source']
    agents = initDict['DATA']['agents']
    outcome = initDict['DATA']['outcome']
    treatment = initDict['DATA']['treatment']

    # Checks.
    assert (isinstance(source, str))
    
    if agents is not None:
        assert (isinstance(agents, int))
        assert (agents > 0)
    
    for obj in [outcome, treatment]:
        assert (isinstance(obj, int))
        assert (obj >= 0)
        
    # Implications
    assert (outcome != treatment)
        
    # Finishing
    return True 


def _check_rho(initDict):
    """ Check RHO block.
    """
    assert (isinstance(initDict, dict))

    # Check keys
    keys = set(['treated', 'untreated'])
    
    assert (keys == set(initDict['RHO'].keys()))
        
    # Distribute elements.
    rho_u1_v = initDict['RHO']['treated']
    rho_u0_v = initDict['RHO']['untreated']

    # Checks.
    for obj in [rho_u1_v, rho_u0_v]:
        value = obj['value']
        is_free = obj['free']
        
        assert (isinstance(value, float))
        assert (-1.00 < value < 1.00)
        assert (is_free in [True, False])
        
    # Finishing.
    return True


def _check_estimation(initDict):
    """ Check ESTIMATION block.
    """
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Check keys.
    keys = set(['algorithm', 'maxiter', 'start', 'gtol', 'epsilon', \
                'asymptotics', 'hessian', \
                'draws', 'alpha', 'differences', 'version'])

    assert (keys == set(initDict['ESTIMATION'].keys()))
    
    # Distribute elements.
    algorithm = initDict['ESTIMATION']['algorithm']
    maxiter = initDict['ESTIMATION']['maxiter']
    start = initDict['ESTIMATION']['start']
    gtol = initDict['ESTIMATION']['gtol']
    epsilon = initDict['ESTIMATION']['epsilon']
    asymptotics = initDict['ESTIMATION']['asymptotics']
    hessian = initDict['ESTIMATION']['hessian']
    draws = initDict['ESTIMATION']['draws']
    alpha = initDict['ESTIMATION']['alpha']
    differences = initDict['ESTIMATION']['differences']
    
    # Checks
    assert (start in ['manual', 'auto'])
    
    assert (algorithm in ['bfgs', 'powell'])
    
    if maxiter is not None:
        
        assert (isinstance(maxiter, int))
        assert (maxiter >= 0)
    
    for obj in [gtol, epsilon]:
        
        assert (isinstance(obj, float))
        assert (obj > 0)

    for obj in [asymptotics]:
        
        assert (obj in [True, False])
    
    assert (hessian in ['numdiff', 'bfgs'])
    
    for obj in [draws]:
        
        assert (isinstance(obj, int))
        assert (obj > 0)
    
    assert (isinstance(alpha, float))
    assert (0 < alpha < 1.00)
    
    assert (differences in ['one-sided', 'two-sided'])
    
    # Implications.
    if algorithm == 'powell':
        assert (hessian == 'numdiff')

    if (maxiter == 0) and (asymptotics is True):
        assert (hessian == 'numdiff')

    # Finishing
    return True


def _check_simulation(initDict):
    """ Check SIMULATION block.
    '"""
    # Antibugging
    assert (isinstance(initDict, dict))
    
    # Check keys
    keys = set(['agents', 'seed', 'target'])

    assert (keys == set(initDict['SIMULATION'].keys()))
    
    # Distribute elements
    agents = initDict['SIMULATION']['agents']
    seed = initDict['SIMULATION']['seed']
    target = initDict['SIMULATION']['target']
    
    # Checks
    assert (isinstance(agents, int))
    assert (agents > 0)
    
    assert (isinstance(seed, int)) 
    assert (seed > 0)
    
    assert (isinstance(target, str))
    
    # Finishing
    return True