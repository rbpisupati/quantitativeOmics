import scipy as sp
import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt
import logging
import sys

log = logging.getLogger(__name__)

def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

### Implemented from limix plot
def _qqplot_bar(M=1000000, alphaLevel=0.05, distr='log10'):
    """calculate theoretical expectations for qqplot"""
    mRange = 10**(sp.arange(sp.log10(0.5), sp.log10(M - 0.5) +
                            0.1, 0.1))  # should be exp or 10**?
    numPts = len(mRange)
    betaalphaLevel = sp.zeros(numPts)  # down in the plot
    betaOneMinusalphaLevel = sp.zeros(numPts)  # up in the plot
    betaInvHalf = sp.zeros(numPts)
    for n in range(numPts):
        m = mRange[n]  # numPLessThanThresh=m
        betaInvHalf[n] = st.beta.ppf(0.5, m, M - m)
        betaalphaLevel[n] = st.beta.ppf(alphaLevel, m, M - m)
        betaOneMinusalphaLevel[n] = st.beta.ppf(1 - alphaLevel, m, M - m)
    betaDown = betaInvHalf - betaalphaLevel
    betaUp = betaOneMinusalphaLevel - betaInvHalf
    theoreticalPvals = mRange / M
    return(betaUp, betaDown, theoreticalPvals)


def qqplot(pv, outPlot, color="#2c7fb8", label='unknown', alphaLevel=0.05):
    distr='log10'
    ax = plt.gca()
    #if (len(pv.shape) == 1) or ((len(pv.shape) == 2) and pv.shape[1] == 1):
#        die("qqplot requires a 1D array of p-values")
    tests = pv.shape[0]
    pnull = (0.5 + sp.arange(tests)) / tests
    Ipv = sp.argsort(pv)

    if distr == 'log10':
        qnull = -sp.log10(pnull)
        qemp = -sp.log10(pv[Ipv])
        xl = '-log10(P) observed'
        yl = '-log10(P) expected'

    plt.plot(qnull, qemp, '.', color=color, label=label)
    # plt.plot([0,qemp.m0x()], [0,qemp.max()],'r')
    plt.plot([0, qnull.max()], [0, qnull.max()], 'r')
    plt.ylabel(xl)
    plt.xlabel(yl)
    if alphaLevel is not None:
        if distr == 'log10':
            betaUp, betaDown, theoreticalPvals = _qqplot_bar(
                M=tests, alphaLevel=alphaLevel, distr=distr)
            lower = -sp.log10(theoreticalPvals - betaDown)
            upper = -sp.log10(theoreticalPvals + betaUp)
            plt.fill_between(-sp.log10(theoreticalPvals),
                             lower, upper, color='grey', alpha=0.5)
            # plt.plot(-sp.log10(theoreticalPvals),lower,'g-.')
            # plt.plot(-sp.log10(theoreticalPvals),upper,'g-.')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    fig = ax.get_figure()
    fig.savefig(outPlot)


def plt_gwas_peaks_matrix(x_ind, y_ind, tair10, plt_color="#d8b365", hist_color="#8c510a"):
    ## given indices for x axis and y axis. this function plots peak matrix
    # tair10 is a class function giving information about the genome eg. bshap.core.the1001g.ArabidopsisGenome
    import seaborn as sns
    sns.set(style="white", color_codes=True)
    p = sns.jointplot(x_ind, y_ind, stat_func = None, marginal_kws={"bins": np.linspace(1, tair10.chr_inds[-1], 250),"color": hist_color }, xlim = (1, tair10.chr_inds[-1]), ylim = (1, tair10.chr_inds[-1]), kind = "scatter", color = plt_color, alpha = 0.1, joint_kws={"s": 8})
    p.ax_marg_y.remove()
    for i in range(len(tair10.chrs)):
        p.ax_joint.plot( (tair10.chr_inds[i + 1] , tair10.chr_inds[i + 1]) , (1,tair10.chr_inds[-1]), '-', color = "gray")
        p.ax_joint.plot( (1,tair10.chr_inds[-1]), (tair10.chr_inds[i + 1] , tair10.chr_inds[i + 1]), '-', color = "gray")
        p.ax_joint.plot( (1,tair10.chr_inds[-1]), (tair10.chr_inds[i] + tair10.cetro_mid[i] , tair10.chr_inds[i] + tair10.cetro_mid[i]), ':k', color = "gray")
        p.ax_joint.plot( (tair10.chr_inds[i] + tair10.cetro_mid[i] , tair10.chr_inds[i] + tair10.cetro_mid[i]), (1,tair10.chr_inds[-1]), ':k', color = "gray")
    p.ax_joint.set_xticks( tair10.chr_inds[0:5] + (np.array(tair10.golden_chrlen)/2) )
    p.ax_joint.set_xticklabels( tair10.chrs )
    p.ax_joint.set_yticks( tair10.chr_inds[0:5] + (np.array(tair10.golden_chrlen)/2) )
    p.ax_joint.set_yticklabels( tair10.chrs )
    p.ax_joint.set_xlabel( "SNPs" )
    p.ax_joint.set_ylabel( "genes" )
    return(p)


def generate_manhattanplot(x_ind, y_ind, tair10, plt_color="#2c7fb8"):
    y_lim_max = np.nanmax(y_ind) + 10
    q = plt.scatter(x_ind, y_ind, color = plt_color)
    for ix in  tair10.chr_inds:
        q.axes.plot( [ix,ix], [0, y_lim_max], 'k-', color = '#bdbdbd')
    q.axes.set_xticks( tair10.chr_inds[0:5] + (np.array(tair10.golden_chrlen)/2) )
    q.axes.set_xticklabels( tair10.chrs )
    q.axes.set_xlabel( "markers" )
    q.axes.set_ylim( [0, y_lim_max] )
    q.axes.set_xlim( [0, tair10.chr_inds[-1]] )
    return(q)
