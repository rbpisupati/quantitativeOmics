## Kinship calculations
import numpy as np
import scipy as sp
import scipy.stats as st
import logging


log = logging.getLogger(__name__)
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
