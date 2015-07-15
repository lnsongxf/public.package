""" Module that holds the effects class.
"""
# standard library
import numpy as np
import random
import scipy

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class EffectCls(metaCls):
    """ Lightweight class for construction of the marginal effect parameters.
    """
    def __init__(self):
        
        self.isLocked = False
    
    def get_effects(self, model_obj, paras_obj, args):
        """ Get effects.
        """
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isinstance(model_obj, modelCls))
        assert (model_obj.getStatus() == True)
        assert (isinstance(paras_obj, parasCls))
        assert (paras_obj.getStatus() == True)

        # Distribute class attributes.
        x_ex_post_eval = model_obj.getAttr('xExPostEval')
        z_eval = model_obj.getAttr('zEval')
        c_eval = model_obj.getAttr('cEval')

        # Marginal benefit of treatment.
        rho_u1_v = paras_obj.getParameters('rho', 'U1,V')
        rho_u0_v = paras_obj.getParameters('rho', 'U0,V')

        coeffs_bene_ex_post = paras_obj.getParameters('bene', 'exPost')
        coeffs_cost = paras_obj.getParameters('cost', None)
        coeffs_choc = paras_obj.getParameters('choice', None)

        sd_v = paras_obj.getParameters('sd', 'V')
        sd_u1 = paras_obj.getParameters('sd', 'U1')
        sd_u0 = paras_obj.getParameters('sd', 'U0')

        bmte_level = np.dot(coeffs_bene_ex_post, x_ex_post_eval)
        smte_level = np.dot(coeffs_choc, z_eval)
        cmte_level = np.dot(coeffs_cost, c_eval)

        bmte_ex_post = np.tile(np.nan, 99)
        cmte_ex_post = np.tile(np.nan, 99)
        smte_ex_ante = np.tile(np.nan, 99)

        eval_points = np.round(np.arange(0.01, 1.0, 0.01), decimals=2)
        quantiles = scipy.stats.norm.ppf(eval_points, loc=0, scale=sd_v)

        # Construct marginal benefit of treatment (ex post)
        bmte_slopes = ((sd_u1/sd_v)*rho_u1_v - (sd_u0/sd_v)*rho_u0_v)*quantiles
        smte_slopes = -quantiles
        cmte_slopes = (((sd_u1/sd_v)*rho_u1_v - (sd_u0/sd_v)*rho_u0_v) +
                       1.0)*quantiles

        # Construct marginal benefit of treatment (ex post)
        for i in range(99):

            bmte_ex_post[i] = bmte_level + bmte_slopes[i]

        # Construct marginal surplus of treatment (ex ante).
        for i in range(99):

            smte_ex_ante[i] = smte_level + smte_slopes[i]

        # Construct marginal cost of treatment (ex ante).
        for i in range(99):

            cmte_ex_post[i] = cmte_level + cmte_slopes[i]

        if args['which'] == 'bmteExPost':

            rslt = bmte_ex_post

        elif args['which'] == 'cmteExAnte':

            rslt = cmte_ex_post

        elif args['which'] == 'smteExAnte':

            rslt = smte_ex_ante

        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (99, ))

        # Finishing.
        return rslt
