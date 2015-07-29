""" Module that checks the user input after processing of the initialization
    file.
"""

# standard library
import numpy as np

''' Main function.
'''


def check_input(init_dict):
    """ Check the input from the initialization file.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))

    # Check keys.
    keys = set(
        ['DATA', 'BENE', 'COST', 'RHO', 'ESTIMATION', 'SIMULATION', 'DERIV'])

    assert (keys == set(init_dict.keys()))

    ''' Check subgroups.
    '''
    _check_eqn(init_dict)

    _check_data(init_dict)

    _check_estimation(init_dict)

    _check_simulation(init_dict)

    _check_rho(init_dict)

    _check_msc(init_dict)

    _check_deriv(init_dict)

    # Finishing
    return True


''' Private auxiliary functions.
'''


def _check_deriv(init_dict):
    """ Check derived information.
    """
    assert (isinstance(init_dict, dict))

    ''' Positions.
    '''
    # Distribute
    all_ = init_dict['DERIV']['pos']['all']

    max_ = init_dict['DERIV']['pos']['max']

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
        list_ = init_dict['DERIV']['excl_bene'][info]['pos']

        num = init_dict['DERIV']['excl_bene'][info]['num']

        assert (isinstance(list_, list))
        assert (isinstance(num, int))
        assert (all(isinstance(pos, np.int64) for pos in list_))
        assert (num >= 0)

    # Cost shifters
    list_ = init_dict['DERIV']['excl_cost']['pos']
    num = init_dict['DERIV']['excl_cost']['num']

    assert (isinstance(list_, list))
    assert (isinstance(num, int))

    assert (all(isinstance(pos, np.int64) for pos in list_))
    assert (num >= 0)

    # Common elements.
    list_ = init_dict['DERIV']['common']['pos']

    pos = init_dict['DERIV']['common']['num']

    assert (isinstance(list_, list))
    assert (isinstance(num, int))

    assert (all(isinstance(pos, np.int64) for pos in list_))
    assert (num >= 0)

    # Finishing.
    return True


def _check_msc(init_dict):
    """ Check selected additional constraints.
    """
    assert (isinstance(init_dict, dict))

    ''' Check that the covariates in the treated and untreated state are
        identical.
    '''
    treated = set(init_dict['BENE']['TREATED']['coeffs']['pos'])
    untreated = set(init_dict['BENE']['UNTREATED']['coeffs']['pos'])

    assert (treated == untreated)

    treated = init_dict['BENE']['TREATED']['coeffs']['info']
    untreated = init_dict['BENE']['UNTREATED']['coeffs']['info']
    num_info = len(treated)

    assert (all((treated[i] == untreated[i]) for i in range(num_info)))

    ''' Check that there are no duplicates in either benefit or cost equation.
    '''
    treated = init_dict['BENE']['TREATED']['coeffs']['pos']
    untreated = init_dict['BENE']['UNTREATED']['coeffs']['pos']
    cost = init_dict['BENE']['UNTREATED']['coeffs']['pos']

    for obj in [treated, untreated, cost]:
        assert (len(obj) == len(set(obj)))

    ''' Check identification.
    '''
    is_free = init_dict['COST']['sd']['free'][0]

    if is_free:
        assert (init_dict['DERIV']['excl_bene']['ex_ante']['num'] > 0)

    ''' Check that outcome and treated are not columns in either benefit or
        cost equations.
    '''
    treated = set(init_dict['BENE']['TREATED']['coeffs']['pos'])
    untreated = set(init_dict['BENE']['UNTREATED']['coeffs']['pos'])
    cost = set(init_dict['BENE']['UNTREATED']['coeffs']['pos'])

    y = set([init_dict['DATA']['outcome']])
    d = set([init_dict['DATA']['treatment']])

    for obj in [treated, untreated, cost]:
        assert (len(obj.intersection(y)) == 0)
        assert (len(obj.intersection(d)) == 0)

    ''' Check that at least one regressor in choice, either ex ante benefit
        or cost shifters.
    '''
    bene = len(init_dict['BENE']['TREATED']['coeffs']['pos'])
    cost = len(init_dict['COST']['coeffs']['pos'])

    no_benefit_shifters = (bene == 0)
    no_cost_shifters = (cost == 0)

    if no_benefit_shifters:
        assert (cost > 0)

    if no_cost_shifters:
        info = sum(init_dict['BENE']['TREATED']['coeffs']['info'])
        assert (info > 0)

    ''' Check that no covariates that are unknown to the agent at the time of
        the treatment decision are specified as cost shifters.
    '''
    info = np.array(init_dict['BENE']['TREATED']['coeffs']['info'])

    benefit_shifters = (len(info) > 0)

    if benefit_shifters:
        pos = np.array(init_dict['BENE']['TREATED']['coeffs']['pos'])
        unknown = set(pos[info == False])
        pos = set(init_dict['COST']['coeffs']['pos'])

        assert (len(unknown.intersection(pos)) == 0)

    # Finishing
    return True


def _check_eqn(init_dict):
    """ Check EQN block.
    """
    assert (isinstance(init_dict, dict))

    # Check keys.
    assert (set(['TREATED', 'UNTREATED']) == set(init_dict['BENE'].keys()))

    ''' BENE.
    '''
    for group in ['TREATED', 'UNTREATED']:

        for subgroup in ['coeffs', 'sd', 'int']:

            positions = None

            values = None

            infos = None

            free = None

            if subgroup == 'coeffs':
                positions = init_dict['BENE'][group][subgroup]['pos']

                values = init_dict['BENE'][group][subgroup]['values']

                infos = init_dict['BENE'][group][subgroup]['info']

                free = init_dict['BENE'][group][subgroup]['free']

            if subgroup in ['sd', 'int']:
                values = init_dict['BENE'][group][subgroup]['values']

                free = init_dict['BENE'][group][subgroup]['free']

            # Position
            if positions is not None:
                assert (isinstance(positions, list))
                assert (all(isinstance(pos, int) for pos in positions))
                assert ((pos >= 0) for pos in positions)

            # Values
            if values is not None:
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
            positions = init_dict['COST'][subgroup]['pos']
            values = init_dict['COST'][subgroup]['values']
            free = init_dict['COST'][subgroup]['free']

        if subgroup in ['sd', 'int']:
            values = init_dict['COST'][subgroup]['values']
            free = init_dict['COST'][subgroup]['free']

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


def _check_data(init_dict):
    """ Check DATA block.
    """
    assert (isinstance(init_dict, dict))

    # Check keys.
    keys = set(['source', 'agents', 'outcome', 'treatment'])

    assert (keys == set(init_dict['DATA'].keys()))

    # Distribute elements.
    source = init_dict['DATA']['source']
    agents = init_dict['DATA']['agents']
    outcome = init_dict['DATA']['outcome']
    treatment = init_dict['DATA']['treatment']

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


def _check_rho(init_dict):
    """ Check RHO block.
    """
    assert (isinstance(init_dict, dict))

    # Check keys
    keys = set(['treated', 'untreated'])

    assert (keys == set(init_dict['RHO'].keys()))

    # Distribute elements.
    rho_u1_v = init_dict['RHO']['treated']
    rho_u0_v = init_dict['RHO']['untreated']

    # Checks.
    for obj in [rho_u1_v, rho_u0_v]:
        value = obj['value']
        is_free = obj['free']

        assert (isinstance(value, float))
        assert (-1.00 < value < 1.00)
        assert (is_free in [True, False])

    # Finishing.
    return True


def _check_estimation(init_dict):
    """ Check ESTIMATION block.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))

    # Check keys.
    keys = set(['algorithm', 'maxiter', 'start', 'gtol', 'epsilon', \
                'asymptotics', 'hessian', \
                'draws', 'alpha', 'differences', 'version'])

    assert (keys == set(init_dict['ESTIMATION'].keys()))

    # Distribute elements.
    algorithm = init_dict['ESTIMATION']['algorithm']
    maxiter = init_dict['ESTIMATION']['maxiter']
    start = init_dict['ESTIMATION']['start']
    gtol = init_dict['ESTIMATION']['gtol']
    epsilon = init_dict['ESTIMATION']['epsilon']
    asymptotics = init_dict['ESTIMATION']['asymptotics']
    hessian = init_dict['ESTIMATION']['hessian']
    draws = init_dict['ESTIMATION']['draws']
    alpha = init_dict['ESTIMATION']['alpha']
    differences = init_dict['ESTIMATION']['differences']

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


def _check_simulation(init_dict):
    """ Check SIMULATION block.
    '"""
    # Antibugging
    assert (isinstance(init_dict, dict))

    # Check keys
    keys = set(['agents', 'seed', 'target'])

    assert (keys == set(init_dict['SIMULATION'].keys()))

    # Distribute elements
    agents = init_dict['SIMULATION']['agents']
    seed = init_dict['SIMULATION']['seed']
    target = init_dict['SIMULATION']['target']

    # Checks
    assert (isinstance(agents, int))
    assert (agents > 0)

    assert (isinstance(seed, int))
    assert (seed > 0)

    assert (isinstance(target, str))

    # Finishing
    return True
