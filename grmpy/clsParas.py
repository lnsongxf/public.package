""" Module that holds the parasCls, which manages all things related to the
    parameter management.
"""
# standard library
import numpy as np

# project library
from grmpy.clsMeta import MetaCls
from grmpy.clsModel import ModelCls


class ParasCls(MetaCls):
    """ Class for the parameter management.
    """

    def __init__(self, model_obj):

        # Antibugging 
        assert (isinstance(model_obj, ModelCls))
        assert (model_obj.get_status() is True)

        # Attach attributes
        self.attr = dict()

        self.attr['para_objs'] = []
        self.attr['num_paras'] = 0
        self.attr['num_free'] = 0
        self.attr['factor'] = None

        self.attr['num_covars_excl_bene_ex_ante'] = model_obj.get_attr(
            'num_covars_excl_bene_ex_ante')
        self.attr['num_covars_excl_cost'] = model_obj.get_attr(
            'num_covars_excl_cost')

        self.attr['without_prediction'] = model_obj.get_attr(
            'without_prediction')
        self.attr['surp_estimation'] = model_obj.get_attr('surp_estimation')

        self.attr['X_ex_ante'] = model_obj.get_attr('X_ex_ante')
        self.attr['X_ex_post'] = model_obj.get_attr('X_ex_post')

        self.attr['num_agents'] = model_obj.get_attr('num_agents')

        # Initialization
        self.attr['model_obj'] = model_obj

        self.isFirst = True

        # Status
        self.is_locked = False

    def add_parameter(self, type_, subgroup, value, is_free, bounds, col):
        """ Add parameters to class instance.
        """
        # Antibugging. 
        assert (is_free in [True, False])
        assert (len(bounds) == 2)
        assert (type_ in ['outc', 'cost', 'sd', 'rho'])
        if type_ in ['outc', 'cost']:
            assert (isinstance(col, int) or (col == 'int'))
        if type_ == 'sd':
            assert (bounds[0] is not None)
            assert (bounds[0] > 0.0)
            assert (col is None)
        if type_ == 'rho':
            assert (bounds[0] > -1.00)
            assert (bounds[1] < 1.00)
            assert (col is None)

        # Initialize parameters.
        count = self.attr['num_paras']

        para_obj = _ParaContainer()

        if is_free:
            id_ = self.attr['num_free']
            para_obj.set_attr('id', id_)
        else:
            para_obj.set_attr('id', None)

        para_obj.set_attr('col', col)
        para_obj.set_attr('count', count)
        para_obj.set_attr('type', type_)
        para_obj.set_attr('subgroup', subgroup)
        para_obj.set_attr('value', value)
        para_obj.set_attr('is_free', is_free)
        para_obj.set_attr('bounds', bounds)
        para_obj.lock()

        # Update class attributes.
        self.attr['para_objs'].append(para_obj)
        self.attr['num_paras'] += 1

        if is_free:
            self.attr['num_free'] += 1

    def get_parameter(self, count):
        """ Get a single parameter object identified by its count. It is 
            important to note, that the selection mechanism refers to all
            parameters not just the true ones. 
        """
        # Antibugging 
        assert (self.get_status() is True)
        assert (count < self.get_attr('num_paras'))

        # Algorithm 
        para_objs = self.get_attr('para_objs')

        for para_obj in para_objs:
            para_count = para_obj.get_attr('count')
            if para_count == count:
                return para_obj

        # Finishing 
        assert (False is True)

    def get_parameters(self, type_, subgroup, is_obj=False):
        """ Get parameter groups.
        """
        # Antibugging 
        assert (self._check_request(type_, subgroup, is_obj) is True)

        # Collect request 
        rslt_list = []

        if type_ in ['outc', 'cost', 'rho', 'sd']:
            for para_obj in self.attr['para_objs']:
                if para_obj.get_attr('type') != type_:
                    continue

                if para_obj.get_attr('subgroup') != subgroup:
                    continue

                if is_obj:
                    rslt_list.append(para_obj)
                else:
                    rslt_list.append(para_obj.get_attr('value'))

        # Special types: Covariances 
        if type_ == 'cov':
            var_one = subgroup.split(',')[0]
            var_two = subgroup.split(',')[1]
            rho = self.get_parameters('rho', var_one + ',' + var_two)
            sd_one = self.get_parameters('sd', var_one)
            sd_two = self.get_parameters('sd', var_two)

            rslt_list.append(rho * sd_one * sd_two)

        # Special types: Variances 
        if type_ == 'var':
            sd_one = self.get_parameters('sd', subgroup)
            rslt_list.append(sd_one ** 2)

        if type_ == 'bene':
            assert (subgroup in ['exPost', 'exAnte'])

            if subgroup == 'exPost':
                outc_treated = self.get_parameters('outc', 'treated')
                outc_untreated = self.get_parameters('outc', 'untreated')

                rslt_list = outc_treated - outc_untreated

            elif subgroup == 'exAnte':

                coeffs_bene_ex_ante = self._prediction_step()

                rslt_list = coeffs_bene_ex_ante

        if type_ == 'choice':
            num_covars_excl_cost = self.get_attr('num_covars_excl_cost')
            num_covar_excl_bene_ex_ante = self.get_attr(
                'num_covars_excl_bene_ex_ante')

            coeffs_bene_ex_ante = self.get_parameters('bene', 'exAnte')
            coeffs_cost = self.get_parameters('cost', None)

            coeffs_bene = np.concatenate(
                (coeffs_bene_ex_ante, np.tile(0.0, num_covars_excl_cost)))
            coeffs_cost = np.concatenate(
                (np.tile(0.0, num_covar_excl_bene_ex_ante), coeffs_cost))

            rslt_list = coeffs_bene - coeffs_cost

        # Dealing with objects 
        if is_obj and type_ in ['rho', 'sd', 'var', 'cov']:
            return rslt_list[0]

        # Type conversion 
        rslt = np.array(rslt_list[:])

        if type_ in ['rho', 'sd', 'var', 'cov']:
            rslt = np.array(rslt[0])

        # Quality check 
        if is_obj is False:
            assert (isinstance(rslt, np.ndarray))
            assert (np.all(np.isfinite(rslt)))

        # Finishing.
        return rslt

    def get_values(self, version, which):
        """ Get all free parameter values.
        """
        # Antibugging  
        assert (self.get_status() is True)
        assert (self._check_integrity() is True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])

        # Main algorithm 
        para_objs = self.attr['para_objs']

        rslt = []

        for para_obj in para_objs:

            is_fixed = (para_obj.get_attr('is_free') is False)

            if is_fixed and (which == 'free'):
                continue

            value = para_obj.get_attr('value')

            if version == 'external':
                value = self._transform_to_external(para_obj,
                                                    para_obj.get_attr('value'))

            rslt.append(value)

        # Type conversion
        rslt = np.array(rslt)

        # Quality checks
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')

        if which == 'all':
            assert (rslt.shape == (self.attr['num_paras'],))
        else:
            assert (rslt.shape == (self.attr['num_free'],))

        # Finishing.
        return rslt

    """ All methods related to updating the parameters. 
    """

    def update(self, x, version, which):
        """ Update all free parameters.
        """
        # Antibugging
        assert (self.get_status() is True)
        assert (self._check_integrity() is True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])

        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')

        if which == 'all':
            assert (x.shape == (self.get_attr('num_paras'),))
        else:
            assert (x.shape == (self.get_attr('num_free'),))

        # Distribute class attributes
        para_objs = self.get_attr('para_objs')

        counter = 0

        for para_obj in para_objs:

            is_fixed = (para_obj.get_attr('is_free') is False)

            if is_fixed and (which == 'free'):
                continue

            value = x[counter]

            if para_obj.get_attr('has_bounds') and (version == 'external'):
                value = self._transform_to_internal(para_obj, value)

            para_obj.set_value(value)

            counter += 1

        # Finishing 
        return True

    def _transform_to_external(self, para_obj, internal_value):
        """ Transform internal values for external use by maximization 
            routine.
        """
        # Antibugging 
        assert (isinstance(para_obj, _ParaContainer))
        assert (para_obj.get_status() is True)
        assert (isinstance(internal_value, float))
        assert (np.isfinite(internal_value))

        # Auxiliary objects 
        lower_bound, upper_bound = para_obj.get_attr('bounds')

        has_lower_bound = (lower_bound is not None)
        has_upper_bound = (upper_bound is not None)

        # Stabilization
        internal_value = self._clip_internal_value(para_obj, internal_value)

        # Upper bound only
        if (not has_lower_bound) and has_upper_bound:
            external_value = np.log(upper_bound - internal_value)

        # Lower bound only
        elif has_lower_bound and (not has_upper_bound):
            external_value = np.log(internal_value - lower_bound)

        # Upper and lower bounds
        elif has_lower_bound and has_upper_bound:
            interval = upper_bound - lower_bound
            transform = (internal_value - lower_bound) / interval
            external_value = np.log(transform / (1.0 - transform))

        # No bounds.
        else:
            external_value = internal_value

        # Quality Check.
        assert (isinstance(external_value, float))
        assert (np.isfinite(external_value))

        # Finishing.
        return external_value

    def _transform_to_internal(self, para_obj, external_value):
        """ Transform external values to internal para_obj.
        """
        # Antibugging 
        assert (isinstance(para_obj, _ParaContainer))
        assert (para_obj.get_status() is True)
        assert (isinstance(external_value, float))
        assert (np.isfinite(external_value))

        # Auxiliary objects
        lower_bound, upper_bound = para_obj.get_attr('bounds')
        has_bounds = para_obj.get_attr('has_bounds')

        has_lower_bound = (lower_bound is not None)
        has_upper_bound = (upper_bound is not None)

        # Stabilization
        if has_bounds:
            external_value = np.clip(external_value, None, 10)

        # Upper bound only
        if (not has_lower_bound) and has_upper_bound:
            internal_value = upper_bound - np.exp(external_value)

        # Lower bound only
        elif has_lower_bound and (not has_upper_bound):
            internal_value = lower_bound + np.exp(external_value)

        # Upper and lower bounds
        elif has_lower_bound and has_upper_bound:
            interval = upper_bound - lower_bound
            internal_value = lower_bound + interval / (
                1.0 + np.exp(-external_value))

            # No bounds
        else:
            internal_value = external_value

        # Stabilization
        internal_value = self._clip_internal_value(para_obj, internal_value)

        # Quality Check
        assert (isinstance(internal_value, float))
        assert (np.isfinite(internal_value))

        # Finishing.
        return internal_value

    @staticmethod
    def _clip_internal_value(para_obj, internal_value):
        """ Assure that internal value not exactly equal to bounds.
        """
        # Antibugging.
        assert (isinstance(para_obj, _ParaContainer))
        assert (para_obj.get_status() is True)
        assert (isinstance(internal_value, float))
        assert (np.isfinite(internal_value))

        # Auxiliary objects
        lower_bound, upper_bound = para_obj.get_attr('bounds')
        has_lower_bound = (lower_bound is not None)
        has_upper_bound = (upper_bound is not None)

        # Check bounds
        if has_lower_bound:
            if internal_value == lower_bound:
                internal_value += 0.01

        if has_upper_bound:
            if internal_value == upper_bound:
                internal_value -= 0.01

        # Quality Check
        assert (isinstance(internal_value, float))
        assert (np.isfinite(internal_value))

        if has_lower_bound:
            assert (lower_bound < internal_value)
        if has_upper_bound:
            assert (upper_bound > internal_value)

        # Finishing.
        return internal_value

    """ Additional private methods.
    """

    def _prediction_step(self):
        """ Prediction step to account for benefit shifters unknown to the agent
            at the time of treatment decision. 
        """
        # Antibugging
        assert (self.get_status() is True)

        # Distribute class attributes
        without_prediction = self.get_attr('without_prediction')
        coeffs_bene_ex_post = self.get_parameters('bene', 'exPost')

        # Check applicability
        if without_prediction:
            return coeffs_bene_ex_post

        x_ex_post = self.get_attr('X_ex_post')
        x_ex_ante = self.get_attr('X_ex_ante')

        # Construct index
        idx_bene = np.dot(coeffs_bene_ex_post, x_ex_post.T)

        if self.attr['factor'] is None:
            pinv = np.linalg.pinv(np.dot(x_ex_ante.T, x_ex_ante))
            self.attr['factor'] = np.dot(pinv, x_ex_ante.T)

        rslt = np.dot(self.attr['factor'], idx_bene)

        # Type conversion
        rslt = np.array(rslt)

        # Quality checks
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')

        # Finishing 
        return rslt

    """ Check integrity of class instance and attribute requests.
    """

    @staticmethod
    def _check_request(type_, subgroup, obj):
        """ Check the validity of the parameter request.
        """
        # Check type
        assert (type_ in ['outc', 'cost', 'rho', 'sd',
                          'var', 'cov', 'bene', 'choice'])

        # Check object
        assert (obj in [True, False])

        if obj:
            assert (type_ in ['outc', 'cost', 'rho', 'sd'])

            # Check subgroup
        if subgroup is not None:
            assert (isinstance(subgroup, str))

        # Finishing
        return True

    @staticmethod
    def _check_integrity():

        return True


""" Private methods and classes of the module.
"""


class _ParaContainer(MetaCls):
    """ Container for parameter class.
    """
    counter = 0

    def __init__(self):
        """ Parameter initialization.
        """

        # Attach attributes
        self.attr = dict()

        self.attr['id'] = None
        self.attr['col'] = None

        self.attr['count'] = None

        self.attr['bounds'] = (None, None)

        self.attr['subgroup'] = None
        self.attr['type'] = None
        self.attr['value'] = None
        self.attr['is_free'] = None

        self.attr['pvalue'] = None
        self.attr['confi'] = (None, None)

        self.attr['has_bounds'] = False

        self.is_locked = False

    """ Public get/set methods.
    """

    def set_value(self, arg):
        """ Set value of parameter object.
        """
        # Antibugging
        assert (isinstance(arg, float))

        # Distribute class attributes.
        is_free = self.attr['is_free']

        value = self.attr['value']

        has_bounds = self.attr['has_bounds']

        lower, upper = self.attr['bounds']

        # Checks
        if not is_free:
            assert (value == arg)

        if has_bounds:
            if lower is not None:
                assert (value > lower)

            if upper is not None:
                assert (value < upper)

        # Set attribute
        self.attr['value'] = arg

    def set_attr(self, key, arg):
        """ Set attribute.
        
            Development Note:
            
                This function overrides the metaCls method. Otherwise, 
                the updating step during estimation is too tedious.
        
        """
        # Antibugging
        assert (self.check_key(key) is True)

        # Set attribute
        self.attr[key] = arg

    """ Private methods
    """

    def derived_attributes(self):
        """ Update endogenous attributes.
        """

        if np.any(self.attr['bounds']) is not None:
            self.attr['has_bounds'] = True

    def _check_integrity(self):
        """ Check integrity.
        """
        # type
        assert (self.get_attr('type') in ['outc', 'cost', 'rho', 'sd'])

        # column
        if self.get_attr('col') is not None:
            col = self.get_attr('col')

            assert (isinstance(col, int) or (col == 'int'))
            if col != 'int':
                assert (col >= 0)

        # subgroup
        if self.get_attr('subgroup') is not None:
            assert (isinstance(self.get_attr('subgroup'), str))

        # value
        assert (isinstance(self.get_attr('value'), float))
        assert (np.isfinite(self.get_attr('value')))

        # is_free
        assert (self.get_attr('is_free') in [True, False])

        # has_bounds
        assert (isinstance(self.attr['bounds'], tuple))
        assert (len(self.attr['bounds']) == 2)

        # value
        assert (isinstance(self.attr['value'], float))
        assert (np.isfinite(self.attr['value']))

        # confi
        assert (isinstance(self.attr['confi'], tuple))
        assert (len(self.attr['confi']) == 2)

        # pvalue
        if self.attr['pvalue'] is not None:
            assert (isinstance(self.attr['pvalue'], float))
            assert (np.isfinite(self.attr['pvalue']))

        # isLocked
        assert (self.is_locked in [True, False])

        # Finishing.
        return True
