""" Module contains all functions required to generate a model object
    from the processed initialization file.
"""

# standard library
import numpy as np

# project library
from grmpy.clsModel import modelCls
from grmpy.tools.msc import *


''' Main function.
'''
def construct_model(init_dict):
    """ Create  model object based on dictionary that contains the information
        from the initialization file.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    
    # Distribute initialization file
    num_covars_excl_bene_ex_ante = init_dict['DERIV']['excl_bene']['ex_ante']['num']

    num_covars_excl_bene_ex_post = init_dict['DERIV']['excl_bene']['ex_post']['num']
    
    num_covars_excl_cost = init_dict['DERIV']['excl_cost']['num']
    
    # Construct data array
    dataset = _process_dataset(init_dict)
    
    rslt = create_matrices(dataset, init_dict)

    # Initialize model object
    model_obj = modelCls()
    
    model_obj.set_attr('Y', rslt['Y'])
    
    model_obj.set_attr('D', rslt['D'])
    
    model_obj.set_attr('x_ex_post', rslt['x_ex_post'])
    
    model_obj.set_attr('x_ex_ante', rslt['x_ex_ante'])
        
    model_obj.set_attr('G', rslt['G'])
    
    model_obj.set_attr('Z', rslt['Z'])

    model_obj.set_attr('num_covars_excl_bene_ex_post',
                      num_covars_excl_bene_ex_post)
    
    model_obj.set_attr('num_covars_excl_bene_ex_ante',
                      num_covars_excl_bene_ex_ante)
    
    model_obj.set_attr('num_covars_excl_cost', num_covars_excl_cost)

    model_obj.set_attr('algorithm', init_dict['ESTIMATION']['algorithm'])

    model_obj.set_attr('epsilon', init_dict['ESTIMATION']['epsilon'])

    model_obj.set_attr('differences', init_dict['ESTIMATION']['differences'])

    model_obj.set_attr('gtol', init_dict['ESTIMATION']['gtol'])

    model_obj.set_attr('maxiter', init_dict['ESTIMATION']['maxiter'])

    model_obj.set_attr('with_asymptotics', init_dict['ESTIMATION']['asymptotics'])

    model_obj.set_attr('numDraws', init_dict['ESTIMATION']['draws'])

    model_obj.set_attr('version', init_dict['ESTIMATION']['version'])

    model_obj.set_attr('hessian', init_dict['ESTIMATION']['hessian'])

    model_obj.set_attr('alpha', init_dict['ESTIMATION']['alpha'])

    model_obj.lock()
    
    # Finishing.
    return model_obj

''' Private auxiliary functions.
'''
def _process_dataset(init_dict):
    """ Processing of dataset by removing missing variables and subset.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
   
    # Distribute initialization file
    file_name = init_dict['DATA']['source']
    
    num_obs = init_dict['DATA']['agents']

    all_pos = init_dict['DERIV']['pos']['all']

    # Load source
    dataset = np.genfromtxt(file_name)

    # Restrict to non-missing
    idx = []
    for i in range(dataset.shape[0]):
        no_missings = np.all(np.isfinite(dataset[i, all_pos]))
        if no_missings:
            idx.append(i)

    dataset = dataset[idx, :]
   
    # Subset selection
    if num_obs is not None:
        dataset = dataset[:num_obs, :]

    # Quality checks
    assert (np.all(np.isfinite(dataset[:, all_pos])))
        
    # Finishing
    return dataset
