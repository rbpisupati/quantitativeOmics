### Linear mixed models using limix
import numpy as np
import scipy as sp
import scipy.stats as st
import h5py as h5
import logging
from . import parsers
from limix.qtl import scan
from . import plot

log = logging.getLogger(__name__)

def get_bh_thres(pvals, fdr_thres=0.05):
    """
    Implements Benjamini-Hochberg FDR threshold (1995)
    adapted from PyGWAS, but changed the thres_pval
    """
    m = len(pvals)
    s_pvals = np.sort(pvals)
    pexpected = (np.arange(1, m + 1)/float(m)) * fdr_thres
    req_inds = np.where(pvals < pexpected)[0]
    if len(req_inds) > 0:
        return(np.amax(pvals[req_inds]))
    else:
        return(0)

def run_lm_st(inputs):
    for snp in inputs.geno.get_snps_iterator(is_chunked=True):
        lm_chunk = scan(np.array(snp[:,inputs.accinds], dtype=int).T, np.array(inputs.pheno.values), test=inputs.test)
        yield(lm_chunk)

def run_lmm_st(inputs):
    for snp in inputs.geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = qtl_test_lmm(np.array(snp[:,inputs.accinds], dtype=int).T, np.array(inputs.pheno.values), inputs.kin, test=inputs.test)
        yield(lmm_chunk)

def run_glmm_st(inputs):
    for snp in inputs.geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = scan(np.array(snp[:,inputs.accinds], dtype=int).T, np.array(inputs.pheno.values), lik = inputs.pheno_type, K = inputs.kin, test=inputs.test, searchDelta=False)
        yield(lmm_chunk)

def getMaf(snp):
    ## Give a 2d array with snps in rows and accs in columns
    alt_freq = np.mean(snp, axis=1)
    ref_freq = 1 - alt_freq
    return(np.minimum.reduce([alt_freq, ref_freq]))

chunk_size = 1000
def lmm_singleTrai(args, maf_thres = 0.05):
    """
    Linear mixed model, association studies
    """
    inputs = parsers.InputsfurLimix(args['genoFile'], args['kinFile'], phenoFile= args['phenoFile'], pheno_type = args['pheno_type'], transform=args['transformation'], test=args['test'])
    if inputs.pheno_type is None:
        lmm = run_lmm_st(inputs)
    else:
        lmm = run_glmm_st(inputs)
    log.info("writing to file: %s" % args['outFile'])
    h5file = h5.File(args['outFile'], 'w')
    if inputs.pheno_type is None:
        h5file.create_dataset('test', compression="gzip", data= "lmm_" + inputs.test, shape=((1,)))
    else:
        h5file.create_dataset('test', compression="gzip", data= "glmm_" + inputs.pheno_type + inputs.test, shape=((1,)))
    h5file.create_dataset('transformation', compression="gzip", data=inputs.pheno.transformation, shape=((1,)))
    h5file.create_dataset('pheno', compression="gzip", data= np.array(inputs.pheno.values), shape=((len(np.array(inputs.pheno.values)),)))
    h5file.create_dataset('chromosomes', compression="gzip", data=np.array(inputs.geno.chromosomes, dtype="int8"), chunks = ((chunk_size,)), shape=(len(inputs.geno.positions),), dtype='int8')
    h5file.create_dataset('positions', compression="gzip", data=inputs.geno.positions, chunks = ((chunk_size,)) , shape=(len(inputs.geno.positions),), dtype='int32')
    h5file.create_dataset('chr_regions', compression="gzip", data=inputs.geno.chr_regions, shape=inputs.geno.chr_regions.shape, dtype='int')
    h5file.create_dataset('chrs', compression="gzip", data=inputs.geno.chrs, shape=inputs.geno.chrs.shape)
    lmm_pvals = h5file.create_dataset('pvalues', compression="gzip", chunks = ((chunk_size,)), shape=(len(inputs.geno.positions),), fillvalue=np.nan, dtype="float64")
    lmm_effsize = h5file.create_dataset('beta_snp', compression="gzip", chunks = ((chunk_size,)), shape=(len(inputs.geno.positions),), fillvalue=np.nan, dtype="float")
    lmm_efferr = h5file.create_dataset('beta_snp_ste', compression="gzip", chunks = ((chunk_size,)), shape=(len(inputs.geno.positions),), fillvalue=np.nan)
    mafs = h5file.create_dataset('maf', compression="gzip", chunks = ((chunk_size,)), shape=(len(inputs.geno.positions),), fillvalue=np.nan)
    index = 0
    for snp in inputs.geno.get_snps_iterator(is_chunked=True):
        lmm_chunk = next(lmm)
        lmm_pvals[index:index+chunk_size] = lmm_chunk.getPv()[0]
        lmm_effsize[index:index+chunk_size] = lmm_chunk.getBetaSNP()[0]
        lmm_efferr[index:index+chunk_size] = lmm_chunk.getBetaSNPste()[0]
        mafs[index:index+chunk_size] = getMaf(snp[:,inputs.accinds])
        index = index + chunk_size
        if index % 50000 == 0:
            log.info("progress: %s positions" % index)
    #h5file.create_dataset('bhy_thres', compression="gzip", data=.transformation, shape=((1,)))
    log.info("generating qqplot!")
    plot.qqplot(np.array(lmm_pvals)[np.where(np.array(mafs) >= maf_thres)[0]], args['outFile'] + ".qqplot.png")
    h5file.close()
    log.info("finished")
