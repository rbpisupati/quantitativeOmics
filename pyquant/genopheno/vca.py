## Variance component analysis
## eQTL
import numpy as np
import scipy as sp
import scipy.stats as st
import h5py
import sys
import logging
import pandas as pd
from pygwas.core import kinship
import parsers
import re

import statsmodels
import limix.modules.varianceDecomposition as var
import limix.modules.qtl as qtl
import limix.io.data as data
import limix.io.genotype_reader as gr
import limix.io.phenotype_reader as phr
import limix.io.data_util as data_util
import limix.utils.preprocess as preprocess
import limix.stats.fdr as FDR

log = logging.getLogger(__name__)
def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def parseBedPosition(g, cisregion):
    # cisregion = Chr1,1,1000
    cisbed = cisregion.split(',')
    if len(cisbed) != 3:
        die("dive a proper bed postion, Ex. Chr1,1,1000")
    reqchrind = np.where(g.chrs == cisbed[0].upper().replace("CHR", ""))[0][0]
    chrpos = g.positions[g.chr_regions[reqchrind][0]:g.chr_regions[reqchrind][1]]
    matchedind = np.sort(np.where(np.in1d(chrpos, np.arange(int(cisbed[1]), int(cisbed[2]))))[0])
    start = matchedind[0] + g.chr_regions[reqchrind][0]
    end = matchedind[-1] + 1
    return (start, end)

def calc_kinship(g, snp_dtype='int8'):
    """
    Calculates IBS kinship
    Forked from the pygwas.core, kinship.calc_ibs_kinship
    But only for the binary file (0, 1, -1), removes all Hets
    """
    num_lines = len(g.accessions)
    #log.info('Allocating matrices for calculation')
    k_mat = sp.zeros((num_lines, num_lines), dtype="uint32")
    log.info('kinship calculation')
    num_snps = sp.zeros((num_lines, num_lines), dtype="uint32")
    snps = g.get_snps_iterator(is_chunked=True, chunk_size=3000)
    chunk_i = 0
    for snp in snps:
        chunk_i += 1
        t_k_mat, t_num_snps = kinship_mat(snp, return_counts=True)
        k_mat = k_mat + t_k_mat
        num_snps = t_num_snps + num_snps
        if chunk_i % 100 == 0:
            log.info("Progress: %s chunks", chunk_i)
            import ipdb; ipdb.set_trace()
        #kin_mat = k_mat / (2 * num_snps) + 0.5
    kin_mat = np.divide(k_mat, num_snps)
    #num_diff_snps = num_snps - k_mat
    return kin_mat

def kinship_mat(snp, return_counts = False, snp_dtype="int8"):
    """
    Kinship for a SNP matrix
    """
    snps_array = sp.array(snp)
    snps_array = snps_array.T
    info_array = sp.mat(np.copy(snps_array).astype(float))
    info_array[info_array >= 0] = 1
    info_array[info_array < 0] = 0
    num_snps = info_array * info_array.T
    snps_array = snps_array.astype(float)
    snps_array[snps_array > 1] = 0.5
    snps_array[snps_array < 0] = 0.5
    sm = sp.mat(snps_array * 2.0 - 1.0)
    k_mat = sm * sm.T
    if return_counts:
        return k_mat, num_snps
    else:
        return np.divide(k_mat, num_snps)


## Single trait variance component analysis
def vca_st(phenoFile, genoFile, cisregion, kinFile=None):
    # local global
    ## Variance component, adapted from Eriko
    ## using limix and pygwas modules
    # packaged on 23.03.2017
    # cisregion = Chr1,1,1000
    geno = parsers.readGenotype(genoFile)
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    (start, end) = parseBedPosition(geno, cisregion)
    cisK = kinship_mat(geno.snps[start:end,:][:,reqAccsInd])
    if kinFile:
        ibs_kin = h5py.File(kinFile)
        transK = np.array(ibs_kin['kinship'])
        transK = transK[reqAccsInd,:][:,reqAccsInd]
    else:
        transK = calc_kinship(geno)
    vc = var.VarianceDecomposition(reqPheno, standardize=True)
    vc.addRandomEffect(K=cisK, normalize=True)
    vc.addRandomEffect(K=transK, normalize=True)
    vc.addRandomEffect(is_noise=True)
    vc.optimize()
    singleVar=vc.getVarianceComps()
    LM=vc.getLML()      #maximum likelihood of this model
    par = singleVar/np.sum(singleVar)*100
    # Results
    log.info("cis effect:   " + str(par[0][0]) + '%')
    log.info("trans effect: " + str(par[0][1]) + '%')
    log.info("noise:        " + str(par[0][2]) + '%')
    return par, LM

def eqtl_st(phenoFile, genoFile, kinFile, test="lrt", covs=None):
    """
    eQTL
    Implemented from limix tutorials
    if kinFile is provided mixed model is performed
    else linear model is utilised

    """
    geno = parsers.readGenotype(genoFile)
    geno_acc = parsers.readGenotype_acc(genoFile)
    if geno_acc is None:
        die("genotype file %s is not provided chunked in columns")
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    snps = sp.mat(geno_acc.snps[:,reqAccsInd], dtype=int).T
    if kinFile:
        ibs_kin = h5py.File(kinFile)
        transK = np.array(ibs_kin['kinship'])
        transK = transK[reqAccsInd,:][:,reqAccsInd]
        model = qtl.test_lmm(snps=snps,pheno=reqPheno,K=transK,covs=covs, test=test)
    else:
        model = qtl.test_lmm(snps=snps,pheno=reqPheno, K=None,covs=covs, test=test)
    pvalues = model.getPv()
    return pvalues
