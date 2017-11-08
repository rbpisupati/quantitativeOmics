## Variance component analysis
## eQTL
import numpy as np
import scipy as sp
import scipy.stats as st
import h5py
import sys
import logging
from . import parsers
from . import kinship
import limix.vardec as var
import limix.qtl as qtl

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
    cisK = kinship.kinship_mat(geno.snps[start:end,:][:,reqAccsInd])
    if kinFile:
        transK = parsers.readKinship(kinFile, reqAccsInd)
    else:
        transK = kinship.calc_kinship(geno)
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
        transK = parsers.readKinship(kinFile, reqAccsInd)
        model = qtl.test_lmm(snps=snps,pheno=reqPheno,K=transK,covs=covs, test=test)
    else:
        model = qtl.test_lmm(snps=snps,pheno=reqPheno, K=None,covs=covs, test=test)
    pvalues = model.getPv()
    return pvalues
