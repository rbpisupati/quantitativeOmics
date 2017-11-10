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
import csv
from pygwas.core import genotype
from pygwas.core import kinship



log = logging.getLogger(__name__)
def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def readPhenoData(phenoFile, geno):
    # use pandas to read the file
    sniffer = csv.Sniffer()
    tpheno = open(phenoFile, 'rb')
    ifheader = sniffer.has_header(tpheno.read(4096))
    sniff_pheno = sniffer.sniff(tpheno.read(4096))
    if ifheader:
        pheno = pd.read_table(phenoFile, header = 0, sep=sniff_pheno.delimiter)
    else:
        raise NotImplementedError
    reqPheno, reqAccsInd = getCommonPheno(pheno, geno)
    return(reqPheno, reqAccsInd)

def getCommonPheno(pheno, geno):
    # taking one temperature into account
    accs_1001g = np.array(geno.accessions)
    reqPheno = []
    reqAccsInd = []
    reqAccs = []
    pheno_accs = np.array(pheno['acc'], dtype="string")
    for i in range(len(accs_1001g)):
        ind = np.where(pheno_accs == accs_1001g[i])[0]
        if len(ind) > 1:
            reqPheno.append(np.mean(pheno['pheno'][ind]))
            reqAccsInd.append(i)
            reqAccs.append(accs_1001g[i])
        elif len(ind) == 1:
            reqPheno.append(pheno['pheno'][ind[0]])
            reqAccsInd.append(i)
            reqAccs.append(accs_1001g[i])
    reqPheno = np.array(reqPheno)
    if len(reqPheno) == 0:
        raise NotImplementedError
    return(reqPheno, reqAccsInd)

H5_EXT = ['.hdf5', '.h5', 'h5py']

def readGenotype(genoFile):
    fileName,fileType = os.path.splitext(genoFile)
    if fileType in H5_EXT:
        g = genotype.load_hdf5_genotype_data(genoFile)
        return(g)
    else:
        raise NotImplementedError

def readGenotype_acc(genoFile):
    # only for ending in hdf5
    fileName,fileType = os.path.splitext(genoFile)
    if os.path.isfile(fileName + '.acc' + fileType):
        g_acc = genotype.load_hdf5_genotype_data(fileName + '.acc' + fileType)
        return(g_acc)
    else:
        return(None)

def readKinship(kinFile, reqAccsInd):
    kinship1001g = h5py.File(kinFile)
    return(kinship1001g['kinship'][reqAccsInd,:][:,reqAccsInd])
