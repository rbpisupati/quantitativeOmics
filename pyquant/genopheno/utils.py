import statsmodels as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import anova
from statsmodels.formula import api as smapi
import numpy as np
import numba as nb
import scipy.stats as stats
import pandas as pd



def marker_to_int(marker_df, marker_ids = ['AA', 'AB', 'BB']):
    if type(marker_df) is pd.Series:
        marker_df_int = pd.Series(np.repeat(np.nan, marker_df.shape), index = marker_df.index )
    elif type(marker_df) is pd.DataFrame:
        marker_df_int = pd.DataFrame(index =  marker_df.index, columns = marker_df.columns)
    for ef_mar_ix in range(len(marker_ids)):
        marker_df_int[ marker_df == marker_ids[ef_mar_ix] ] = ef_mar_ix + 1
    ## gives an error if all the marker IDS are not given
    return(marker_df_int.astype(int))

def marker_to_factor_df(marker_str, marker_ids = ['AA', 'AB', 'BB']):
    marker_df = pd.DataFrame( index = marker_str.index )
    for ef_marker in marker_ids:
        marker_df.loc[:,ef_marker] = marker_str == ef_marker
    return( marker_df )


### Below Numba functions are adapted from
#  https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b
@nb.njit(parallel=True)
def tiecorrect(rankvals):
    """
    parallelized version of scipy.stats.tiecorrect
    """
    tc = np.ones(rankvals.shape[1], dtype=np.float64)
    for j in nb.prange(rankvals.shape[1]):
        arr = np.sort(np.ravel(rankvals[:,j]))
        idx = np.nonzero(
            np.concatenate(
                (
                    np.array([True]),
                    arr[1:] != arr[:-1], 
                    np.array([True])
                )
            )
        )[0]
        cnt = np.diff(idx).astype(np.float64)

        size = np.float64(arr.size)
        if size >= 2:
            tc[j] = 1.0 - (cnt**3 - cnt).sum() / (size**3 - size)

    return tc


@nb.njit(parallel=True)
def rankdata(data):
    """
    parallelized version of scipy.stats.rankdata
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in nb.prange(data.shape[1]):
        arr = np.ravel(data[:, j])
        sorter = np.argsort(arr)

        arr = arr[sorter]
        obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))

        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()

        # cumulative counts of each unique value
        count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
        ranked[:, j] = 0.5 * (count[dense] + count[dense - 1] + 1)

    return ranked


def proportions_ztest( count, nobs ):
    """
    Adapted from statsmodels.stat.proportion proportions_ztest

    Only implemented two-sided test
    """
    k_sample = np.shape(count)
    count = np.asarray(count)
    nobs = np.asarray(nobs)
    
    assert np.shape(count)[0] == np.shape(nobs)[0], "provide arrays with same number of rows"
    assert k_sample[1] == 2, "value must be provided for a 1-sample test"
    
    prop = count * 1. / nobs
    diff = prop[:,0] - prop[:,1]
    
    p_pooled = np.sum(count, axis = 1) * 1. / np.sum(nobs, axis = 1)
    nobs_fact = np.sum(1. / nobs, axis = 1)
    var_ = p_pooled * (1 - p_pooled) * nobs_fact
    std_diff = np.sqrt(var_)
    
    np_zstat = diff / std_diff
    np_pval = stats.norm.sf(np.abs(np_zstat))*2
    
    return((np_zstat, np_pval))



def mannwhitneyu(x, y, use_continuity=True):
    """Version of Mann-Whitney U-test that runs in parallel on 2d arrays
    
    this is the two-sided test, asymptotic algo only
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[1] == y.shape[1]

    n1 = x.shape[0]
    n2 = y.shape[0]

    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1, :]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y
    T = tiecorrect(ranked)

    # if *everything* is identical we'll raise an error, not otherwise
    if np.all(T == 0):
        raise ValueError('All numbers are identical in mannwhitneyu')
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)

    meanrank = n1 * n2 / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (bigu - meanrank) / sd

    p = np.clip(2 * stats.norm.sf(z), 0, 1)

    return u2, p
