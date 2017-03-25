"""
Data parsers
"""

import numpy as np
import scipy as st
import pandas as pd
import h5py
import sys
import logging
import os.path
from pygwas.core import genotype



log = logging.getLogger(__name__)
def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def readPhenoData(phenoFile, geno):
    # use pandas to read the file
    pheno = pd.read_table(phenoFile)
    reqPheno, reqAccsInd = getCommonPheno(pheno, geno)
    return reqPheno, reqAccsInd

def getCommonPheno(pheno, geno):
    # taking one temperature into account
    accs_1001g = np.array(geno.accessions)
    reqPheno = []
    reqAccsInd = []
    reqAccs = []
    for i in range(len(accs_1001g)):
        ind = np.where(pheno['acc']  == accs_1001g[i] + "_10C" )[0]
        if len(ind) > 0:
            reqPheno.append(pheno['pheno'][ind[0]])
            reqAccsInd.append(i)
            reqAccs.append(accs_1001g[i])
    reqPheno = np.array(reqPheno)
    return reqPheno, reqAccsInd

H5_EXT = ['.hdf5', '.h5', 'h5py']

def readGenotype(genoFile):
    fileName,fileType = os.path.splitext(genoFile)
    if fileType in H5_EXT:
        g = genotype.load_hdf5_genotype_data(genoFile)
        return g
    else:
        die("For now provide hdf5 to load in pygwas")

def readGenotype_acc(genoFile):
    # only for ending in hdf5
    fileName,fileType = os.path.splitext(genoFile)
    if os.path.isfile(fileName + '.acc' + fileType):
        g_acc = genotype.load_hdf5_genotype_data(fileName + '.acc' + fileType)
        return g_acc
    else:
        return None
