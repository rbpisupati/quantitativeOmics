"""
Data parsers
""""
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
    log.info("loading phenotype file")
    sniffer = csv.Sniffer()
    tpheno = open(phenoFile, 'rb')
    ifheader = sniffer.has_header(tpheno.read(4096))
    tpheno.seek(0)
    sniff_pheno = sniffer.sniff(tpheno.read(4096))
    if ifheader:
        pheno = pd.read_table(phenoFile, header = 0, sep=sniff_pheno.delimiter)
    else:
        pheno = pd.read_table(phenoFile, header = None, sep=sniff_pheno.delimiter)
    pheno.columns = np.array(['acc','pheno'])
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
        log.info("loading genotype file")
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
    log.info("loading kinship file")
    kinship1001g = h5py.File(kinFile)
    return(kinship1001g['kinship'][reqAccsInd,:][:,reqAccsInd])

class InputsfurLimix(object):

    def __init__(self, genoFile, phenoFile, kinFile, pheno_type, transform, test=None):
        self.geno = readGenotype(genoFile)
        self.pheno_type = pheno_type
        self.test = test
        self.pheno = self.parse_pheno(phenoFile, transform)
        self.parseKinFile(kinFile)

    def parse_pheno(self, phenoFile, transform):
        from pygwas.core import phenotype
        reqPheno, self.accinds = readPhenoData(phenoFile, self.geno)
        pheno = phenotype.Phenotype(self.geno.accessions[self.accinds], reqPheno, None)
        if transform is not None:
            pheno.transform(transform)
        return(pheno)

    def parseKinFile(self, kinFile):
        if kinFile is not None:
            self.kin = readKinship(kinFile, self.accinds)
        else:
            self.kin = kinship.calc_kinship(self.geno)
