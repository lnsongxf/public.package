#!/usr/bin/env python
# wscript

# standard library
import os
import glob
import shutil
import fnmatch

# build directories
top = '.'
out = '.bld'

def options(opt):
    
    opt.add_option('--test', \
        action  = 'store_true', \
        dest    = 'test', \
        default = False, \
        help    = 'Execute unit testing library.')    
    
def configure(conf):

    conf.env.project_paths = {}

    conf.env.project_paths['MAIN'] = os.getcwd()
    
    conf.env.project_paths['GRM_TOOLBOX'] = conf.env.project_paths['MAIN']

    tools_dir = conf.env.project_paths['GRM_TOOLBOX'] + '/tools'

    conf.load('runPyScript', tooldir = tools_dir)

def build(bld):
    
    # Distribute options.
    test  = bld.options.test
    
    #Change directory.
    os.chdir(bld.env.project_paths['GRM_TOOLBOX'])
    
    # Fix permissions.
    set_permissions()
    
    bld.env.PROJECT_PATHS = set_project_paths(bld)

    bld.add_group() 
    
    if(test):
        
        bld.recurse('tests')    

def distclean(ctx):
    
    #/* manual clean    */

    remove_filetypes_distclean('.')
    
    remove_for_distclean('.waf-1.6.4-8c7ad4bb8e1ca65b04e5d8dd9d0dac54')

    remove_for_distclean('.bld')

''' Auxiliary functions.
'''
def set_permissions():
    ''' Set permissions.
    '''
    
    files = glob.glob('scripts/*.py')

    for file_ in files:
        
        os.chmod(file_, 0777)
        
def remove_for_distclean(path):
    ''' Remove path, where path can be either a directory or a file. The
        appropriate function is selected. Note, however, that if an 
        OSError occurs, the function will just path.

    '''

    if os.path.isdir(path):

        shutil.rmtree(path)
    
    if os.path.isfile(path):

        os.remove(path)

def remove_filetypes_distclean(path):
    ''' Remove nuisance files from the directory tree.

    '''

    matches = []

    for root, _, filenames in os.walk('.'):

        for filetypes in ['*.aux','*.out','*.log','*.pyc', '*.so', '*~', \
                          '*tar', '*.zip', '.waf*', '*lock*', '*.mod', '*.a', \
                          '*.grm.*']:

                for filename in fnmatch.filter(filenames, filetypes):
                    
                    matches.append(os.path.join(root, filename))

    matches.append('.lock-wafbuild')

    for files in matches:

        remove_for_distclean(files)

def set_project_paths(ctx):
    ''' Return a dictionary with project paths represented by Waf nodes. This is
        required such that the run_py_script works as the whole PROJECT_ROOT is
        added to the Python path during execution.

    ''' 

    pp = {}

    pp['PROJECT_ROOT'] = '.'
   
    for key, val in pp.items():

        pp[key] = ctx.path.make_node(val)
   
    return pp

