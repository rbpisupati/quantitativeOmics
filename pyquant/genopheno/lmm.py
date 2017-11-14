### Linear mixed models using limix
import numpy as np
import scipy as sp
import scipy.stats as st
import h5py as h5
import sys
import logging
import pandas as pd
from . import parsers
from . import kinship as qkin
import re
from pygwas.core import phenotype
import matplotlib.pyplot as plt

from limix.qtl import qtl_test_lmm

log = logging.getLogger(__name__)

def run_lmm_st(geno, reqPheno, reqKinship, reqAccsInd):
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = qtl_test_lmm(np.array(snp[:,reqAccsInd], dtype=int).T, reqPheno, reqKinship)
        yield(lmm_chunk)

chunk_size = 1000
def lmm_singleTrai(phenoFile, genoFile, kinFile, outFile, transformation = "None", test = "lrt"):
    """
    Linear mixed model, association studies
    """
    log.info("loading genotype file")
    geno = parsers.readGenotype(genoFile)
    log.info("loading phenotype file")
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    phenos = phenotype.Phenotype(geno.accessions[reqAccsInd], reqPheno, None)
    phenos.transform(transformation)
    import ipdb; ipdb.set_trace()
    log.info("loading kinship file")
    reqKinship = parsers.readKinship(kinFile, reqAccsInd)
    log.info("done")
    log.info("writing to file: %s" % outFile)
    h5file = h5.File(outFile, 'w')
    h5file.create_dataset('test', compression="gzip", data=test, shape=((1,)))
    h5file.create_dataset('transformation', compression="gzip", data=phenos.transformation, shape=((1,)))
    h5file.create_dataset('positions', compression="gzip", data=geno.positions, chunks = ((chunk_size,)) , shape=(len(geno.positions),), dtype='int32')
    h5file.create_dataset('chr_regions', compression="gzip", data=geno.chr_regions, shape=geno.chr_regions.shape, dtype='int')
    h5file.create_dataset('chrs', compression="gzip", data=geno.chrs, shape=geno.chrs.shape)
    lmm_pvals = h5file.create_dataset('pvalues', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm_effsize = h5file.create_dataset('beta_snp', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm_efferr = h5file.create_dataset('beta_snp_ste', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    mafs = h5file.create_dataset('maf', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm = run_lmm_st(geno, phenos.values, reqKinship, reqAccsInd)
    index = 0
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = next(lmm)
        lmm_pvals[index:index+chunk_size] = lmm_chunk.getPv()[0]
        lmm_effsize[index:index+chunk_size] = lmm_chunk.getBetaSNP()[0]
        lmm_efferr[index:index+chunk_size] = lmm_chunk.getBetaSNPste()[0]
        mafs[index:index+chunk_size] = np.mean(snp, axis=1)
        index = index + chunk_size
        if divmod(index, chunk_size)[1] % 100 == 0:
            log.info("progress: %s positions" % index)
    h5file.close()
