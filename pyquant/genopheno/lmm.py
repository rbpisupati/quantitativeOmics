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

def lmm_singleTrai(phenoFile, genoFile, kinFile, test = "lrt"):
    """
    Linear mixed model, association studies
    """
    geno = parsers.readGenotype(genoFile)
    geno_acc = parsers.readGenotype_acc(genoFile)
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    reqKinship = parsers.readKinship(kinFile, reqAccsInd)
    for snp in geno.get_snps_iterator(is_chunked=True):
        import pdb; pdb.set_trace()
        lmm = qtl_test_lmm(snp[reqAccsInd], reqPheno, reqKinship)
