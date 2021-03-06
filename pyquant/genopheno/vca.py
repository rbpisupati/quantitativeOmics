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
    if int(cisbed[1]) < int(cisbed[2]):
        matchedind = np.where( (chrpos >= int(cisbed[1])) & (chrpos <= int(cisbed[2])) )[0]
    else:
        matchedind = np.where( (chrpos <= int(cisbed[1])) & (chrpos >= int(cisbed[2])) )[0]
    return( matchedind + g.chr_regions[reqchrind][0] )

## Single trait variance component analysis
def vca_st(args):
    import limix.vardec as var
    from pyquant.mixmogam import linear_models
    from limix.heritability import estimate
    # local global
    # packaged on 23.03.2017
    # cisregion = Chr1,1,1000
    inputs = parsers.InputsfurLimix(args['genoFile'], args['kinFile'], phenoFile = args['phenoFile'], pheno_type = None, transform=args['transformation'])
    log.info("calculating cisK")
    cispos_ix = parseBedPosition(inputs.geno, args['cisregion'])
    cisK = kinship.kinship_mat(inputs.geno.snps[cispos_ix,:][:,inputs.accinds])
    log.info("running lmm")
    ## From mixmogam
    #par = linear_models.local_vs_global_mm(np.array(inputs.pheno.values), cisK, inputs.kin, inputs.kin, h0_res=None)
    # we need to ask regarding the output of this function
    # From limix
    vc = var.VarianceDecomposition(np.array(inputs.pheno.values), standardize=False)
    vc.addFixedEffect()
    vc.addRandomEffect(K=cisK)
    vc.addRandomEffect(K=inputs.kin)
    vc.addRandomEffect(is_noise=True)
    import ipdb; ipdb.set_trace()
    #h2 = estimate(np.array(inputs.pheno.values), 'normal', inputs.kin, verbose=False)
    vc.optimize()
    singleVar=vc.getVarianceComps()
    LM=vc.getLML()      #maximum likelihood of this model
    par = singleVar/np.sum(singleVar)*100
    ## Results
    log.info("cis effect:   " + str(par[0][0]) + '%')
    log.info("trans effect: " + str(par[0][1]) + '%')
    log.info("noise:        " + str(par[0][2]) + '%')
    return par

def eqtl_st(phenoFile, genoFile, kinFile, test="lrt", covs=None):
    import limix.qtl as qtl
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
