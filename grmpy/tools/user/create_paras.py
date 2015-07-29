""" Module contains all functions required to generate a parameter object
    from the processed initialization file.
"""

# standard library
import sys

import statsmodels.api as sm
import numpy as np


# project library.
from grmpy.clsParas import ParasCls
from grmpy.clsModel import ModelCls

""" Main function.
"""


def construct_paras(init_dict, model_obj, is_simulation):
    """ Construct parameter object.
    """
    # Antibugging 
    assert (isinstance(init_dict, dict))
    assert (model_obj.get_status() is True)

    # Distribute auxiliary objects
    start = init_dict['ESTIMATION']['start']

    # Initialize with manual starting values
    paras_obj = _initialize_parameters(init_dict, model_obj)

    # Update with automatic starting values
    if start == 'auto' and (not is_simulation):
        paras_obj = _auto_start(paras_obj, model_obj)

    # Quality
    assert (paras_obj.get_status() is True)

    # Finishing
    return paras_obj


""" Private auxiliary functions.
"""


def _initialize_parameters(init_dict, model_obj):
    """ Get starting values from initialization file.
    """

    def _get_values(group, subgroup, init_dict):
        """ Order the starting values such that they are matched with the correct
            columns. This includes the intercept.
        """

        def _collect_information(positions, dict_):
            """ Order the information appropriately.
            """
            # Antibugging 
            assert (isinstance(positions, list))
            assert (isinstance(dict_, dict))
            assert (set(dict_.keys()) == set(positions))

            # Initialize containers 
            values = []
            is_frees = []
            cols = []

            # Collect info
            for pos in positions:
                values += [dict_[pos]['value']]
                is_frees += [dict_[pos]['is_free']]
                cols += [dict_[pos]['col']]

            # Quality
            assert all(isinstance(value, float) for value in values)
            assert all(isinstance(is_free, bool) for is_free in is_frees)

            # Finishing
            return values, cols, is_frees

        # Antibugging
        assert (isinstance(init_dict, dict))
        assert (group in ['BENE', 'COST'])

        if group == 'BENE':
            assert (subgroup in ['TREATED', 'UNTREATED'])

        if group == 'COST':
            assert (subgroup is None)

        # Distribute information.
        common = init_dict['DERIV']['common']['pos']

        excl_bene_ex_post = init_dict['DERIV']['excl_bene']['ex_post']['pos']

        excl_cost = init_dict['DERIV']['excl_cost']['pos']

        # Initialize container
        dict_ = dict()

        # Benefits
        positions = None

        if group == 'BENE':

            # Coefficients.
            is_frees = init_dict['BENE'][subgroup]['coeffs']['free'][:]

            values = init_dict['BENE'][subgroup]['coeffs']['values'][:]

            positions = init_dict['BENE'][subgroup]['coeffs']['pos'][:]

            for pos in positions:
                dict_[pos] = {}

                dict_[pos]['value'] = values.pop(0)

                dict_[pos]['is_free'] = is_frees.pop(0)

                dict_[pos]['col'] = pos

            # Intercept.
            dict_['int'] = dict()
            dict_['int']['value'] = init_dict[
                'BENE'][subgroup]['int'][
                'values'][0]
            dict_['int']['is_free'] = \
                init_dict['BENE'][subgroup]['int']['free'][0]

            dict_['int']['col'] = 'int'

            # Collect in order
            positions = excl_bene_ex_post + common + ['int']

        # Costs
        if group == 'COST':

            # Coefficients
            is_frees = init_dict['COST']['coeffs']['free'][:]

            values = init_dict['COST']['coeffs']['values'][:]

            positions = init_dict['COST']['coeffs']['pos'][:]

            for pos in positions:
                dict_[pos] = dict()

                dict_[pos]['value'] = values.pop(0)
                dict_[pos]['is_free'] = is_frees.pop(0)
                dict_[pos]['col'] = pos

            # Intercept.
            dict_['int'] = {}

            dict_['int']['value'] = init_dict['COST']['int']['values'][0]

            dict_['int']['is_free'] = init_dict['COST']['int']['free'][0]

            dict_['int']['col'] = 'int'

            positions = common + ['int'] + excl_cost

        # Create output.
        values, cols, is_frees = _collect_information(positions, dict_)

        # Finishing.
        return values, cols, is_frees

    """ Core function.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    assert (model_obj.get_status() is True)

    # Distribute information
    num_covars_ex_post = len(init_dict['BENE']['TREATED']['coeffs']['pos'])

    num_covars_cost = len(init_dict['COST']['coeffs']['pos'])

    # Initialize parameter container
    paras_obj = ParasCls(model_obj)

    # Benefits

    # Treated
    values, cols, is_frees = _get_values('BENE', 'TREATED', init_dict)

    for i in range(num_covars_ex_post + 1):
        type_ = 'outc'
        subgroup = 'treated'
        value = values[i]
        col = cols[i]

        is_free = is_frees[i]

        paras_obj.add_parameter(type_, subgroup, value, is_free=is_free,
                                bounds=(None, None), col=col)

    # Untreated
    values, cols, is_frees = _get_values('BENE', 'UNTREATED', init_dict)

    for i in range(num_covars_ex_post + 1):
        type_ = 'outc'
        subgroup = 'untreated'
        value = values[i]
        col = cols[i]
        is_free = is_frees[i]
        paras_obj.add_parameter(type_, subgroup, value, is_free=is_free,
                                bounds=(None, None), col=col)

    # Costs
    values, cols, is_frees = _get_values('COST', None, init_dict)

    for i in range(num_covars_cost + 1):
        type_ = 'cost'

        value = values[i]

        col = cols[i]

        is_free = is_frees[i]

        paras_obj.add_parameter(type_, None, value, is_free=is_free,
                                bounds=(None, None), col=col)

    # Correlation parameters
    value = init_dict['RHO']['treated']['value']
    is_free = init_dict['RHO']['treated']['free']

    paras_obj.add_parameter('rho', 'U1,V', value, is_free, (-0.99, 0.99),
                            col=None)

    value = init_dict['RHO']['untreated']['value']
    is_free = init_dict['RHO']['untreated']['free']

    paras_obj.add_parameter('rho', 'U0,V', value, is_free, (-0.99, 0.99),
                            col=None)

    # Disturbances
    value = init_dict['BENE']['UNTREATED']['sd']['values'][0]
    is_free = init_dict['BENE']['UNTREATED']['sd']['free'][0]

    paras_obj.add_parameter('sd', 'U0', value, is_free=is_free,
                            bounds=(0.01, None), col=None)

    value = init_dict['BENE']['TREATED']['sd']['values'][0]
    is_free = init_dict['BENE']['TREATED']['sd']['free'][0]

    paras_obj.add_parameter('sd', 'U1', value, is_free=is_free,
                            bounds=(0.01, None), col=None)

    value = init_dict['COST']['sd']['values'][0]
    is_free = init_dict['COST']['sd']['free'][0]

    paras_obj.add_parameter('sd', 'V', value, is_free=is_free,
                            bounds=(0.01, None), col=None)

    paras_obj.lock()

    # Finishing
    return paras_obj


def _auto_start(paras_obj, model_obj):
    """ Get automatic starting values.
    """

    def _compute_starting_values(model_obj, which):
        """ Get starting values.
        """
        # Antibugging
        assert (model_obj.get_status() is True)
        assert (which in ['treated', 'untreated', 'cost'])

        # Data selection.
        y = model_obj.get_attr('Y')
        d = model_obj.get_attr('D')
        x = model_obj.get_attr('X_ex_post')
        g = model_obj.get_attr('G')

        # Subset selection 
        if which == 'treated':
            y = y[d == 1]
            x = x[(d == 1), :]

        elif which == 'untreated':
            y = y[d == 0]
            x = x[(d == 0), :]

        # Model selection
        coeffs, sd = None, None
        if which in ['treated', 'untreated']:
            ols_rslt = sm.OLS(y, x).fit()
            coeffs = ols_rslt.params
            sd = np.array(np.sqrt(ols_rslt.scale))
        elif which == 'cost':
            stdout_current = sys.stdout
            sys.stdout = open('/dev/null', 'w')
            probit_rslt = sm.Probit(d, g).fit()
            coeffs = -probit_rslt.params
            sd = np.array(1.0)
            sys.stdout = stdout_current

        # Quality checks
        assert (isinstance(coeffs, np.ndarray))
        assert (isinstance(sd, np.ndarray))

        assert (np.all(np.isfinite(coeffs)))
        assert (np.all(np.isfinite(sd)))

        assert (coeffs.ndim == 1)
        assert (sd.ndim == 0)

        assert (coeffs.dtype == 'float')
        assert (sd.dtype == 'float')

        # Type conversions.
        coeffs = coeffs.tolist()
        sd = float(sd)

        # Finishing.
        return coeffs, sd

    """ Core function.
    """
    # Antibugging
    assert (isinstance(paras_obj, ParasCls))
    assert (paras_obj.get_status() is True)

    assert (isinstance(model_obj, ModelCls))
    assert (model_obj.get_status() is True)

    # Benefits 
    for subgroup in ['treated', 'untreated']:

        para_objs = paras_obj.get_parameters('outc', subgroup, is_obj=True)

        coeffs, sd = _compute_starting_values(model_obj, subgroup)

        assert (len(para_objs) == len(coeffs))

        for para_obj in para_objs:

            coeff = coeffs.pop(0)

            # Check applicability
            if not para_obj.get_attr('is_free'):
                continue

            para_obj.set_attr('value', coeff)

        label = 'U1'

        if subgroup == 'untreated':
            label = 'U0'

        para_obj = paras_obj.get_parameters('sd', label, is_obj=True)

        # Check applicability
        if not para_obj.get_attr('is_free'):
            continue

        para_obj.set_attr('value', sd)

    # Cost
    para_objs = paras_obj.get_parameters('cost', None, is_obj=True)

    coeffs, sd = _compute_starting_values(model_obj, 'cost')

    assert (len(para_objs) == len(coeffs))

    for para_obj in para_objs:

        coeff = coeffs.pop(0)

        # Check applicability
        if not para_obj.get_attr('is_free'):
            continue

        para_obj.set_attr('value', coeff)

    para_obj = paras_obj.get_parameters('sd', 'V', is_obj=True)

    para_obj.set_attr('value', sd)

    # Correlations
    for corr in ['U1,V', 'U0,V']:

        para_obj = paras_obj.get_parameters('rho', corr, is_obj=True)

        # Check applicability
        if not para_obj.get_attr('is_free'):
            continue

        para_obj.set_attr('value', 0.0)


    # Quality.
    assert (paras_obj.get_status() is True)

    # Finishing.
    return paras_obj
