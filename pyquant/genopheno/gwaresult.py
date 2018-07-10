# Main module for gwas
# Summary statistics
import logging
import h5py as h5
import numpy as np
import pandas as pd
import os.path
import glob
import sys
import csv
import itertools
from abc import abstractmethod

log = logging.getLogger(__name__)
tair_chrs = ['Chr1','Chr2','Chr3','Chr4','Chr5']
chrslen = [34964571, 22037565, 25499034, 20862711, 31270811]
golden_chrlen = [30427671, 19698289, 23459830, 18585056, 26975502]


def load_gwas_result(hdf5_file):
    return(HDF5GWASRESULT(hdf5_file))


class HDF5GWASRESULT(object):

    def __init__(self,hdf5_file):
        self.h5file = h5.File(hdf5_file,'r')
        self.chr_regions = np.array(self.h5file['chr_regions'])
        self.test = np.array(self.h5file['test'])[0]
        self.transformation = np.array(self.h5file['transformation'])[0]

    def __getattr__(self, name, filter_pos_ix=None):
        if name not in ['positions', 'maf', 'beta_snp', 'pvalues','chrs']:
            raise AttributeError("%s is not in the keys for HDF5. Only accepted values are ['positions', 'maf', 'beta_snp', 'pvalues']" % name)
        if filter_pos_ix is None:
            return(self.h5file[str(name)])
        elif type(filter_pos_ix) is np.ndarray:
            rel_pos_ix = filter_pos_ix - filter_pos_ix[0]
            return(self.h5file[str(name)][filter_pos_ix[0]:filter_pos_ix[-1]+1][rel_pos_ix])
        else:
            return(self.h5file[str(name)][filter_pos_ix])

    def get_chrs(self, filter_pos_ix=None):
        if 'chrs' in self.__dict__.keys():
            if filter_pos_ix is None:
                return( self.chrs )
            return(self.chrs[np.array(filter_pos_ix)])
        chrs = np.repeat('Chr', len(self.positions))
        for ic in range(len(np.array(self.h5file['chrs']))):
            chrs[self.chr_regions[ic][0]:self.chr_regions[ic][1]] = self.h5file['chrs'][ic]
        self.chrs = chrs
        if filter_pos_ix is None:
            return(chrs)
        else:
            return(chrs[np.array(filter_pos_ix)])

    def calc_maf_filter(self, maf_threshold=0.05):
        if 'maf_filter_ix' not in self.__dict__.keys():
            self.maf_filter_ix = np.where( np.array(self.__getattr__("maf", None)) > maf_threshold )[0]

    def calc_qvalues(self, transform_qvalues = True):
        if  "qvalues" not in self.__dict__.keys():
            from limix.stats import fdr
            self.calc_maf_filter()
            pvals = self.__getattr__("pvalues", self.maf_filter_ix)
            pvals[pvals == 0] = np.nan
            self.qvalues, self.pi0 = fdr.qvalues(pvals, m = len(self.positions), return_pi0 = True )
            if transform_qvalues:
                self.logqval = transform(self.qvalues)

    def get_peaks(self, maf_threshold=0.05):
        ## retuns two array of indices, 1 --> fdr as 0.01 2 --> fdr as 0.05
        self.calc_maf_filter(maf_threshold=maf_threshold)
        self.calc_qvalues()
        fdr_5per_ix = np.where(self.qvalues < 0.05)[0]
        fdr_1per_ix = fdr_5per_ix[ np.where(self.qvalues[fdr_5per_ix] < 0.01)[0] ]
        return( [ self.maf_filter_ix[fdr_1per_ix], self.maf_filter_ix[np.setdiff1d(fdr_5per_ix, fdr_1per_ix)] ] )

    def get_chr_positions(self, filter_pos_ix):
        return( pd.DataFrame(np.column_stack((self.get_chrs(filter_pos_ix=filter_pos_ix), self.__getattr__("positions", filter_pos_ix=filter_pos_ix) )), columns=['chr', 'position']) )



def transform(pvals):
    return(-np.log10(pvals))
