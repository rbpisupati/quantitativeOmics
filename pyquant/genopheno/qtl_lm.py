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
            acc_ix = self._matching_accs_ix(genotypes)
            self.genos = np.array( genotypes.__getattr__('value') )[:, acc_ix[0]].T
            self.pheno = self.pheno.iloc[acc_ix[1],]
            assert self.genos.shape[0] == self.pheno.shape[0]
            self.bed_str = pd.Series(genotypes.get_bed_df(None, return_str=True))

    def _matching_accs_ix(self, genotypes):
        ## genotypes is a HDF51001gTable
        common_accs = np.intersect1d( genotypes.accessions, np.array(list(self.pheno.index)) )
        acc_ix_geno = genotypes.get_matching_accs_ix( common_accs )
        acc_ix_pheno = np.where( np.in1d( np.array(list(self.pheno.index)), common_accs ) )[0]
        assert len(acc_ix_geno) == len(acc_ix_pheno)
        assert len(acc_ix_geno) > 0
        if len(acc_ix_pheno) / float(self.pheno.shape[0]) <= 0.2 or len(acc_ix_geno) / float(genotypes.accessions.shape[0]) <= 0.2 :
            log.warn("There are very few samples in genotype that match to samples with phenotypes")
        return((acc_ix_geno, acc_ix_pheno))

    def get_filter_accs_nans(self, filter_thres = 0.1):
        ## Filters accessions which have less then 10% of markers defined
        filter_accs_no_nans = np.where( pd.DataFrame(self.genos.T).isna().sum() < filter_thres * self.genos.shape[1])[0]
        if type(self.pheno) is pd.Series:
            return(np.intersect1d( filter_accs_no_nans, np.where(self.pheno.notna())[0]))
        else:
            return(np.intersect1d( filter_accs_no_nans, np.where(self.pheno.notna().all(axis=1))[0]))

    def filter_markers_HWP(self, pval_thres = 0.05):
        ### Filters markers based on hardy weinberg principle
        mend_stats = np.zeros(0, dtype=float)
        mend_pval = np.zeros(0, dtype=float)
        for e_ix in range(self.genos.shape[1]):
            obs_n = np.array([ len(np.where( self.genos[:,e_ix] == eg )[0]) for eg in range(3) ])
            exp_n = np.array((0.25,0.5,0.25)) * np.sum(obs_n)
            sch = st.chisquare(f_exp=exp_n, f_obs=obs_n)
            mend_stats = np.append(mend_stats, sch.statistic)
            mend_pval = np.append(mend_pval, sch.pvalue)
        return((mend_stats, mend_pval))

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
        #lm.vscore = lm.score(lm.X, m)
        lm.vscore = sklm.explained_variance_score( m, lm.y_pred )
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
            lm = qtl_test_lm(self.genos[filter_nanaccs_ix, :], np.array( self.pheno.iloc[filter_nanaccs_ix] ), covs = covs[filter_nanaccs_ix] )
            if len(np.where(np.isfinite(lm.getPv()[0]))[0]) == 0:
                return(None)
            return(lm)
        else:
            lm = []
            for cl in self.pheno:
                lm.append(qtl_test_lm(self.genos[filter_nanaccs_ix, :], np.array( self.pheno[cl][filter_nanaccs_ix] ), covs = covs[filter_nanaccs_ix] ))
            return(lm) ## returns an array

    def plot_qtl_map(self, lm, tair10, output_file=None, ylim = None):
        from . import plot as pygplot
        ## tair10 is the class variable for the genome
        qvalues = -np.log10(lm.getPv()[0,:])
        q = pygplot.generate_manhattanplot( tair10.get_genomewide_inds( self.bed_str ), qvalues, tair10, ylim = ylim)
        q.axes.set_ylabel("q-values")
        if output_file is not None:
            q.figure.savefig(output_file, height = 50, width = 50)
        return(q)

    def get_qtl_peaks(self, lm, output_file=None, gene_name = None):
        pvals = lm.getPv().flatten()
        betas = lm.beta_snp.flatten()
        pval_nan_ix = np.where(np.isfinite(pvals))[0]
        qvals_lg = -np.log10( lstat.qvalues(pvals[pval_nan_ix]) )
        peak_inds = np.where((qvals_lg > 2) & np.isfinite(qvals_lg) )[0]
        qval_cats = np.around( qvals_lg[peak_inds], decimals = 2)
        peak_strs = self.bed_str[pval_nan_ix[peak_inds]]
        betas_cat = np.around(betas[pval_nan_ix[peak_inds]], decimals = 2)
        peaks_df = pd.DataFrame( np.column_stack((peak_strs, qval_cats, betas_cat)), columns = ["peak_pos", "peak_cat", "peak_beta"] )
        if output_file is not None:
            for ef in range(peaks_df.shape[0]):
                output_file.write( "%s\t%s\t%s\t%s\n" % ( gene_name, str(peaks_df.iloc[ef, 0]), str(peaks_df.iloc[ef, 1]), str(peaks_df.iloc[ef, 2]) ) )
        return( peaks_df )
