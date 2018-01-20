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
from . import plot

log = logging.getLogger(__name__)

def run_lmm_st(geno, reqPheno, reqKinship, reqAccsInd, test):
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = qtl_test_lmm(np.array(snp[:,reqAccsInd], dtype=int).T, reqPheno, reqKinship, test=test)
        yield(lmm_chunk)

def getMaf(snp):
    ## Give a 2d array with snps in rows and accs in columns
    alt_freq = np.mean(snp, axis=1)
    ref_freq = 1 - alt_freq
    return(np.minimum.reduce([alt_freq, ref_freq]))

chunk_size = 1000
def lmm_singleTrai(phenoFile, genoFile, kinFile, outFile, transformation = "None", test = "lrt", maf_thres = 0.05):
    """
    Linear mixed model, association studies
    """
    geno = parsers.readGenotype(genoFile)
    reqPheno, reqAccsInd = parsers.readPhenoData(phenoFile, geno)
    phenos = phenotype.Phenotype(geno.accessions[reqAccsInd], reqPheno, None)
    if transformation is not None:
        phenos.transform(transformation)
    reqKinship = parsers.readKinship(kinFile, reqAccsInd)
    log.info("writing to file: %s" % outFile)
    h5file = h5.File(outFile, 'w')
    h5file.create_dataset('test', compression="gzip", data=test, shape=((1,)))
    h5file.create_dataset('transformation', compression="gzip", data=phenos.transformation, shape=((1,)))
    h5file.create_dataset('chromosomes', compression="gzip", data=np.array(geno.chromosomes, dtype="int8"), chunks = ((chunk_size,)), shape=(len(geno.positions),), dtype='int8')
    h5file.create_dataset('positions', compression="gzip", data=geno.positions, chunks = ((chunk_size,)) , shape=(len(geno.positions),), dtype='int32')
    h5file.create_dataset('chr_regions', compression="gzip", data=geno.chr_regions, shape=geno.chr_regions.shape, dtype='int')
    h5file.create_dataset('chrs', compression="gzip", data=geno.chrs, shape=geno.chrs.shape)
    lmm_pvals = h5file.create_dataset('pvalues', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm_effsize = h5file.create_dataset('beta_snp', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm_efferr = h5file.create_dataset('beta_snp_ste', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    mafs = h5file.create_dataset('maf', compression="gzip", chunks = ((chunk_size,)), shape=(len(geno.positions),), fillvalue=np.nan)
    lmm = run_lmm_st(geno, np.array(phenos.values), reqKinship, reqAccsInd, test)
    index = 0
    for snp in geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = next(lmm)
        lmm_pvals[index:index+chunk_size] = lmm_chunk.getPv()[0]
        lmm_effsize[index:index+chunk_size] = lmm_chunk.getBetaSNP()[0]
        lmm_efferr[index:index+chunk_size] = lmm_chunk.getBetaSNPste()[0]
        mafs[index:index+chunk_size] = getMaf(snp[:,reqAccsInd])
        index = index + chunk_size
        if index % 50000 == 0:
            log.info("progress: %s positions" % index)
    log.info("generating qqplot!")
    plot.qqplot(np.array(lmm_pvals)[np.where(np.array(mafs < maf_thres))[0]], outFile + ".qqplot.png")
    h5file.close()
    log.info("finished")
