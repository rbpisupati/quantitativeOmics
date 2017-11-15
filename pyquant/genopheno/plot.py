import scipy as sp
import scipy.stats as st
import sys
import matplotlib.pylab as plt
import logging

log = logging.getLogger(__name__)


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
