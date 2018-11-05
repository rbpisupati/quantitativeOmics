## Linear model for QTL mapping and eQTL
## eQTL
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st
import h5py
import sys
import logging

from bshap.core import the1001g
from sklearn.linear_model import LinearRegression
from limix.qtl import qtl_test_lm
import limix.stats as lstat

log = logging.getLogger(__name__)
def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)


class QTLmapperLM(object):
    """
    Inputs
    1. Genotype array, np ndarray (rows as accs)
    2. Phenotype data, pandas dataframe (row index as accs names)

    """
    def __init__(self, genotypes, phenos, bed_str = None ):
        assert type(phenos) is pd.DataFrame or type(phenos) is pd.Series
        self.pheno = phenos
        if type(genotypes) is np.ndarray:
            assert genotypes.shape[0] == self.pheno.shape[0]
            self.genos = genotypes
            self.bed_str = bed_str
            if bed_str is None:
                log.warn( "please provide bed_str (array with genotype marker positions), useful in plotting" )
        elif type(genotypes) is the1001g.HDF51001gTable:
            acc_ix = genotypes.get_matching_accs_ix( list(self.pheno.index) )
            assert len(acc_ix) > 0
            self.genos = np.array( genotypes.__getattr__('value') )[:, acc_ix].T
            assert self.genos.shape[0] == self.pheno.shape[0]
            self.bed_str = pd.Series(genotypes.get_bed_df(None, return_str=True))


    def get_filter_accs_nans(self, filter_thres = 0.1):
        ## Filters accessions which have less then 10% of markers defined
        filter_accs_no_nans = np.where(np.unique(np.where(np.isnan(self.genos))[0], return_counts=True)[1] < filter_thres * self.genos.shape[1])[0]
        if type(self.pheno) is pd.Series:
            return(np.intersect1d( filter_accs_no_nans, np.where(self.pheno.notna())[0]))
        else:
            return(np.intersect1d( filter_accs_no_nans, np.where(self.pheno.notna().all(axis=1))[0]))

    def filter_markers_HWP(self, filter_thres = 0.1):
        ### Filters markers based on hardy weinberg principle
        return(None)

    def skip_mapping(self, min_accs = 20):
        ## Function to check if the phenotype has not many nans
        filter_nanaccs_ix = self.get_filter_accs_nans()
        if len(filter_nanaccs_ix) < min_accs:
            return(True)
        return(False)

    def get_geno_meths_df(self, geno_ix):
        if type(self.pheno) is pd.Series:
            geno_pheno = pd.DataFrame(np.column_stack(( self.genos[:,geno_ix], self.pheno )), columns = ['geno', self.pheno.name ])
        else:
            geno_pheno = pd.DataFrame(np.column_stack(( self.genos[:,geno_ix], self.pheno )), columns = np.append('geno', self.pheno.columns.values ) )
        return(geno_pheno)

    def single_loci_linear_model(self, geno_ix, pheno_ix = 0):
        import sklearn.metrics as sklm
        geno_pheno = self.get_geno_meths_df(geno_ix)
        X = np.array([geno_pheno.dropna().iloc[:,0]]).T
        m = np.array([geno_pheno.dropna().iloc[:, pheno_ix + 1]]).T
        lm = LinearRegression().fit(X, m)
        lm.X = X
        lm.y_pred = lm.predict(lm.X)
        lm.mse = sklm.mean_squared_error(m, lm.y_pred)
        lm.vscore = lm.score(lm.X, m)
        sse = np.sum((lm.predict(lm.X) - m) ** 2, axis=0) / float(lm.X.shape[0] - lm.X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(lm.X.T, lm.X))))])
        lmt = lm.coef_ / se
        lm.p = 2 * (1 - st.t.cdf(np.abs(lmt), m.shape[0] - lm.X.shape[1]))
        return(lm)

    def get_qtl_maps(self, covs = None):
        filter_nanaccs_ix = self.get_filter_accs_nans()
        if covs is None:
            covs = np.ones( ( self.genos.shape[0] ) )
        else:  ### Need to also filter the accessions where covs has nan
            assert type(covs) is np.ndarray
            filter_nanaccs_ix = np.intersect1d( filter_nanaccs_ix, np.where(np.isfinite( np.array(covs) ))[0] )
        if type(self.pheno) is pd.Series:
            lm = qtl_test_lm(self.genos[filter_nanaccs_ix, :], np.array( self.pheno[filter_nanaccs_ix] ), covs = covs[filter_nanaccs_ix] )
            if len(np.where(np.isfinite(lm.getPv()[0]))[0]) == 0:
                return(None)
            return(lm)
        else:
            lm = []
            for cl in self.pheno:
                lm.append(qtl_test_lm(self.genos[filter_nanaccs_ix, :], np.array( self.pheno[cl][filter_nanaccs_ix] ), covs = covs[filter_nanaccs_ix] ))
            return(lm) ## returns an array

    def plot_qtl_map(self, lm, tair10, output_file=None):
        from . import plot as pygplot
        ## tair10 is the class variable for the genome
        q = pygplot.generate_manhattanplot( tair10.get_genomewide_inds( self.bed_str ), -np.log10(lm.getPv()[0,:]), tair10)
        q.axes.set_ylabel("p-values")
        if output_file is not None:
            q.figure.savefig(output_file, height = 50, width = 50)
        return(q)

    def get_qtl_peaks(self, lm, output_file=None, req_gene = None):
        pvals = lm.getPv().flatten()
        pval_nan_ix = np.where(np.isfinite(pvals))[0]
        qvals = lstat.qvalues(pvals[pval_nan_ix])
        peak_inds = np.where(qvals < 0.01)[0]
        qval_cats = np.around(-np.log10( qvals[peak_inds] ), decimals = 2)
        #qval_cats = np.array(pd.cut(qvals[peak_inds], bins = [0,0.00000001,0.000001,0.0001,0.001,0.01], labels=["8","6", "4", "3","2"]), dtype="string")
        peak_strs = self.bed_str[pval_nan_ix[peak_inds]]
        peaks_df = pd.DataFrame( np.column_stack((peak_strs, qval_cats)), columns = ["peak_pos", "peak_cat"] )
        if output_file is not None:
            for ef in range(peaks_df.shape[0]):
                output_file.write( "%s\t%s\t%s\n" % ( req_gene, str(peaks_df.iloc[ef, 0]), str(peaks_df.iloc[ef, 1]) ) )
        return( peaks_df )
