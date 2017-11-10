### Linear mixed models using limix
import numpy as np
import scipy as sp
import scipy.stats as st
import h5py
import sys
import logging
import pandas as pd
from . import parsers
from . import kinship as qkin
import re

from limix.qtl import qtl_test_lmm

log = logging.getLogger(__name__)

def run_lmm_st(geno, reqPheno, reqKinship):
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = qtl_test_lmm(np.array(snp[:,reqAccsInd], dtype=int).T, reqPheno, reqKinship)
        yield(lmm_chunk)


def lmm_singleTrai(phenoFile, genoFile, kinFile, test = "lrt"):
    """
    Linear mixed model, association studies
    """
    log.info("loading genotype file")
    geno = parsers.readGenotype(genoFile)
    geno_acc = parsers.readGenotype_acc(genoFile)
    log.info("loading phenotype file")
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    log.info("loading kinship file")
    reqKinship = parsers.readKinship(kinFile, reqAccsInd)
    log.info("done")
    lmm = run_lmm_st(geno, reqPheno, reqKinship)
    import pdb; pdb.set_trace()
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = next(lmm)
