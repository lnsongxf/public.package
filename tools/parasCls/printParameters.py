#!/usr/bin/env python
''' Script to print current instance of parameter class.
'''

# standard library
import os
import sys
import shutil

import cPickle  as pkl
import numpy    as np

# edit pythonpath
dir_ = os.path.realpath(__file__).replace('/tools/parasCls/printParameters.py','')
sys.path.insert(0, dir_)

# project library
import tools.modAuxiliaryFunctions as grmTools

''' Auxiliary functions.
'''
def getInfo(paraObj):
    ''' Get information from parameter object.
    '''
    
    id_      = paraObj.getAttr('id')
    value    = paraObj.getAttr('value')
    startVal = paraObj.getAttr('startVal')
    bounds   = str(paraObj.getAttr('bounds'))
    col      = paraObj.getAttr('col')
    
    # Special treatments.
    if(id_ is None): id_ = '---'

    if(col is None): col = '---'

    if(startVal is None): 
        
        startVal = '---'

    else:
        
        startVal = '{0:5.2f}'.format(startVal)   
        
   
    # Collect.
    rowArgs = [id_, col, startVal, value, bounds]

    return rowArgs

def _writeMainFile():
    
    tex = open('printParameters.grm.tex', 'w')
    
    tex.write(r'''  \input{''' + dir_ + r'''/msc/texHeader.tex} 
                    \begin{document} 
                    
                    \thispagestyle{empty}
                    \setcounter{page}{1}''')

    tex.write(r'\input{t-printParameters}\newpage')

    tex.write(r'\input{t-printPrediction}')

    tex.write(r'''\end{document}''')

    tex.close()

def _tableParameters(parasObj):
    
    tex   = open('t-printParameters.tex', 'w')

    tableHeader = r'''\begin{center}
                \topcaption{Instance of Parameter Class}
                \tablefirsthead{ \hline }
                \tablelasttail{ \hline \multicolumn{7}{l}{\hspace{12pt} \tiny \textbf{Notes}: }}
                \begin{xtabular}{cccccccc}'''
                     
    columnLabel = r''' Count & Label & ID & Column & Start & Value &  Restriction\\\midrule ''' + '''\n'''

    # Format of rows depending on parameter restriction.
    tableRow ='''{0[0]} &{0[1]}  &{0[2]}  & {0[3]}  & {0[4]} & {0[5]:5.2f} &{0[6]}\\tabularnewline\n'''
    
    tableFooter = r'''\bottomrule\end{xtabular}\end{center}'''
                
    tex.write(tableHeader) 
    
    tex.write(r'''\toprule\mc{7}{c}{ Outcome } \\\toprule ''' + '''\n''') 

    idx   = 1
    
    ''' Outcome Treated.
    '''
    tex.write(r'''\midrule\mc{7}{c}{ \textit{Treated} } \\\midrule ''' + '''\n''') 
 
    tex.write(columnLabel) 
    
    parasList = parasObj.getParameters('outc', 'treated', isObj = True)
    
    for para in parasList:
        
        count = para.getAttr('count')
        
        rowArgs = [str(count)] +  [r'$\beta_{1' + ',' + str(idx) + '}$'] + getInfo(para)
            
        tex.write(tableRow.format(rowArgs))
        
        idx   += 1

    ''' Outcome Untreated.
    '''
    idx = 1
    
    tex.write(r'''\midrule\mc{7}{c}{ \textit{Untreated} } \\\midrule ''' + '''\n''') 
 
    tex.write(columnLabel) 
    
    parasList = parasObj.getParameters('outc', 'untreated', isObj = True)

    for para in parasList:
        
        count = para.getAttr('count')
        
        rowArgs = [str(count)] + [r'$\beta_{0' + ',' + str(idx) + '}$'] + getInfo(para)
            
        tex.write(tableRow.format(rowArgs))
        
        idx   += 1
                       
    ''' Cost.
    '''
    idx = 1
    
    tex.write(r'''\toprule\mc{7}{c}{ Cost } \\\toprule ''' + '''\n''') 
 
    tex.write(columnLabel) 
    
    parasList = parasObj.getParameters('cost', None, isObj = True)

    for para in parasList:
        
        count = para.getAttr('count')
        
        rowArgs = [str(count)] + [r'$\gamma_{' + str(idx) + '}$'] + getInfo(para)
            
        tex.write(tableRow.format(rowArgs))

        idx   += 1
    
    ''' Coefficients of correlation.
    '''
    tex.write(r'\toprule\mc{7}{c}{Coefficients of Correlation}\\\toprule')

    tex.write(columnLabel) 
    
    
    paraObj  = parasObj.getParameters('rho', 'U1,V', isObj = True) 
    
    count = paraObj.getAttr('count')
    
    rowArgs = [str(count)] + [r'$\rho_{U_1, V}$'] + getInfo(paraObj)  
            
    tex.write(tableRow.format(rowArgs))
    
    
    paraObj  = parasObj.getParameters('rho', 'U0,V', isObj = True) 
 
    count = paraObj.getAttr('count')
 
    rowArgs = [str(count)] + [r'$\rho_{U_0, V}$'] + getInfo(paraObj)  
            
    tex.write(tableRow.format(rowArgs))
    
    ''' Standard deviations.
    '''
    tex.write(r'\toprule\mc{7}{c}{Standard Deviation}\\\toprule')
    
    tex.write(columnLabel) 
    

    paraObj  = parasObj.getParameters('sd', 'U1', isObj = True)

    count = paraObj.getAttr('count')

    rowArgs = [str(count)] + [r'$\sigma_{U_1}$'] + getInfo(paraObj) 
    
    tex.write(tableRow.format(rowArgs))

    
    paraObj  = parasObj.getParameters('sd', 'U0', isObj = True) 

    count = paraObj.getAttr('count')

    rowArgs = [str(count)] + [r'$\sigma_{U_0}$'] + getInfo(paraObj)  
    
    tex.write(tableRow.format(rowArgs))

    
    paraObj  = parasObj.getParameters('sd', 'V', isObj = True) 

    count = paraObj.getAttr('count')

    rowArgs = [str(count)] + [r'$\sigma_{V}$'] + getInfo(paraObj)
    
    tex.write(tableRow.format(rowArgs))   
    
    ''' Wrapping up.
    '''
    tex.write(tableFooter)    

    tex.close()

def _tablePrediction(parasObj):
    
    tex   = open('t-printPrediction.tex', 'w')
    
    tableHeader = r'''\begin{center}
                    \topcaption{Prediction Model}
                    \tablefirsthead{ \toprule }
                    \tablelasttail{ \toprule \multicolumn{3}{l}{\hspace{12pt} \tiny \textbf{Notes}: }}
                    \begin{xtabular}{ccc}'''
                         
    columnLabel = r''' Count & Label  & Value\\\midrule ''' + '''\n'''
    
    # Format of rows depending on parameter restriction.
    tableRow ='''{0[0]} &{0[1]}  & {0[2]:5.2f}\\tabularnewline\n'''
        
    tableFooter = r'''\end{xtabular}\end{center}'''
                    
    tex.write(tableHeader) 
    
    ''' Writing parameters
    '''
    count = 1
 
    tex.write(columnLabel) 
    
    parasList = parasObj.getParameters('bene', 'exAnte') 

    for para in parasList:
        
        rowArgs = [str(count)] + [r'$(\alpha_{1' + ',' + str(count) + \
                                  r'} - \alpha_{0' + ',' + str(count) + '})$'] \
                                  + [para]
            
        tex.write(tableRow.format(rowArgs))
        
        count   += 1
        
    ''' Wrapping up.
    '''
    tex.write(tableFooter)    
    
    tex.close()

''' Core files.
'''
if(not os.path.exists('grm.rslt')): os.makedirs('grm.rslt')

parasObj = pkl.load(open('parasObj.grm.pkl', 'r'))

hasStep = (os.path.isfile('stepParas.grm.out'))

if(hasStep):
       
    internalValues = np.array(np.genfromtxt('stepParas.grm.out'), dtype = 'float', ndmin = 1)

    parasObj.updateValues(internalValues, isExternal = False, isAll = False)      

os.chdir('grm.rslt')

_writeMainFile()

_tableParameters(parasObj)

_tablePrediction(parasObj)

sys.stdout = open('/dev/null', 'w')   
        
grmTools.runExternalProgramWait('pdflatex', 'printParameters.grm.tex')
    
grmTools.runExternalProgramWait('pdflatex', 'printParameters.grm.tex')
    
sys.stdout = sys.__stdout__

shutil.copy('printParameters.grm.pdf', '../paraClsInstance.grm.pdf')