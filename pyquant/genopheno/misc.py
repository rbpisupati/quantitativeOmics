### Misc
import numpy as np
import pandas as pd

def calculate_ld(mcs_req_position, snps_req_position):
    ### Please give in integers here
    #accepted ones are 0, 1 and -1
    req_pos_haps = np.core.defchararray.add(mcs_req_position.astype("str"), snps_req_position.astype("str"))
    req_pos_haps_ix = np.flatnonzero(np.core.defchararray.find(req_pos_haps,'-')==-1)
    pi_s_0 = len( np.where( snps_req_position[req_pos_haps_ix] == 0 )[0] ) / float(len(req_pos_haps_ix))
    pi_m_0 = len( np.where( mcs_req_position[req_pos_haps_ix] == 0 )[0] ) / float(len(req_pos_haps_ix))
    pi_ms_0 = len( np.where( req_pos_haps[req_pos_haps_ix] == '00' )[0] ) / float(len(req_pos_haps_ix))
    d_ms_0 = pi_ms_0 - (pi_m_0 * pi_s_0)
    deno_r2 = (pi_m_0 * pi_s_0 * (1 - pi_m_0) * (1 - pi_s_0))
    if deno_r2 == 0:
        return(np.nan)
    return((d_ms_0**2) / deno_r2)

class calculate_ld_obj(object):

    def __init__(self, mcs_req_position, geno):
        ## geno == pygwas genotype class
        self.mcs = mcs_req_position
        self.geno = geno

    @staticmethod
    def calculate_ld(mcs_req_position, snps_req_position):
        ### Please give in integers here
        #accepted ones are 0, 1 and -1
        req_pos_haps = np.core.defchararray.add(mcs_req_position.astype("str"), snps_req_position.astype("str"))
        req_pos_haps_ix = np.flatnonzero(np.core.defchararray.find(req_pos_haps,'-')==-1)
        pi_s_0 = len( np.where( snps_req_position[req_pos_haps_ix] == 0 )[0] ) / float(len(req_pos_haps_ix))
        pi_m_0 = len( np.where( mcs_req_position[req_pos_haps_ix] == 0 )[0] ) / float(len(req_pos_haps_ix))
        pi_ms_0 = len( np.where( req_pos_haps[req_pos_haps_ix] == '00' )[0] ) / float(len(req_pos_haps_ix))
        d_ms_0 = pi_ms_0 - (pi_m_0 * pi_s_0)
        deno_r2 = (pi_m_0 * pi_s_0 * (1 - pi_m_0) * (1 - pi_s_0))
        if deno_r2 == 0:
            return(np.nan)
        return((d_ms_0**2) / deno_r2)

    def calculate_ld_pos(self, snps_pos_ix):
        mcs_ld_req_pos = np.zeros(0, dtype=float)
        snp_ld_req_pos = np.zeros(0, dtype=float)
        for i in range(snps_pos_ix[0], snps_pos_ix[1]):
            snps_req_position = self.geno.snps[i, :][accs_snps_ix]
            mcs_ld_req_pos = np.append(mcs_ld_req_pos, self.calculate_ld( self.mcs, snps_req_position ))
            snp_ld_req_pos = np.append(snp_ld_req_pos, self.calculate_ld( self.geno.snps[int(sum(snps_pos_ix)/2), :][accs_snps_ix], snps_req_position ))
        filter_pos_ix = np.where(~np.isnan(mcs_ld_req_pos))[0]
        abs_dist_pos = self.geno.positions[snps_pos_ix[0]:snps_pos_ix[1]]
        #abs_dist_pos = abs_dist_pos - abs_dist_pos % 5
        return((abs_dist_pos[filter_pos_ix], mcs_ld_req_pos[filter_pos_ix], snp_ld_req_pos[filter_pos_ix]))


def calc_ld_table(snps, max_ld_dist=2000, min_r2=0.2, verbose=True, normalize=False):
    """
    Calculate LD between all SNPs using a sliding LD square

    This function only retains r^2 values above the given threshold
    """
    # Normalize SNPs (perhaps not necessary, but cheap)
    if normalize:
        snps = snps.T
        snps = (snps - sp.mean(snps, 0)) / sp.std(snps, 0)
        snps = snps.T


    if verbose:
        print("Calculating LD table")
    t0 = time.time()
    num_snps, num_indivs = snps.shape
    ld_table = {}
    for i in range(num_snps):
        ld_table[i] = {}

    a = min(max_ld_dist, num_snps)
    num_pairs = (a * (num_snps - 1)) - a * (a + 1) * 0.5
    if verbose:
        print("Correlation between %d pairs will be tested" % num_pairs)
    num_stored = 0
    for i in range(0, num_snps - 1):
        start_i = i + 1
        end_i = min(start_i + max_ld_dist, num_snps)
        ld_vec = sp.dot(snps[i], sp.transpose(snps[start_i:end_i])) / float(num_indivs)
        ld_vec = sp.array(ld_vec).flatten()
        for k in range(start_i, end_i):
            ld_vec_i = k - start_i
            if ld_vec[ld_vec_i] ** 2 > min_r2:
                ld_table[i][k] = ld_vec[ld_vec_i]
                ld_table[k][i] = ld_vec[ld_vec_i]
                num_stored += 1
        if verbose:
            if i % 1000 == 0:
                sys.stdout.write('.')
#                 sys.stdout.write('\b\b\b\b\b\b\b%0.2f%%' % (100.0 * (min(1, float(i + 1) / (num_snps - 1)))))
                sys.stdout.flush()
    if verbose:
        sys.stdout.write('Done.\n')
        if num_pairs > 0:
            print("Stored %d (%0.4f%%) correlations that made the cut (r^2>%0.3f)." % (num_stored, 100 * (num_stored / float(num_pairs)), min_r2))
        else:
            print('-')
    t1 = time.time()
    t = (t1 - t0)
    if verbose:
        print("\nIt took %d minutes and %0.2f seconds to calculate the LD table" % (t / 60, t % 60))
    del snps
    return ld_table
