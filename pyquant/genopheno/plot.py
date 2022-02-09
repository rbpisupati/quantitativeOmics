import scipy as sp
import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
import seaborn as sns
sns.set(style="white", color_codes=True)
import math

log = logging.getLogger(__name__)

def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def scale_colors(minval, maxval, val, safe_colors = None):
    import palettable
    if safe_colors is None:
        safe_colors = palettable.colorbrewer.sequential.BuGn_7.colors
    import sys
    EPSILON = sys.float_info.epsilon  # Smallest possible difference.
    i_f = float(val-minval) / float(maxval-minval) * (len(safe_colors)-1)
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    if f < EPSILON:
        ret_col = safe_colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = safe_colors[i], safe_colors[i+1]
        ret_col = int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))
    return('#%02x%02x%02x' % (ret_col[0], ret_col[1], ret_col[2]))

np_scale_colors = np.vectorize(scale_colors)



def millify(n):
    """
    Function to replace 0's on an integer n with Kb or Mb
    """
    millnames = ['','kb','Mb']
    n = float(n)
    millidx = max(0,min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return('{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx]))

### Implemented from limix plot
def _qqplot_bar(M=1000000, alphaLevel=0.05, distr='log10'):
    """
    Calculate theoretical expectations for qqplot
    """
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


def plt_gwas_peaks_matrix(x_ind, y_ind, tair10, peak_heights = None, legend_scale = None, plt_color="#d8b365", hist_color="#8c510a"):
    """
    given indices for x axis and y axis. this function plots our favorite matrix plot
    tair10 is a class function giving information about the genome eg. pyGenome.Genome
    """
    if peak_heights is not None:
        if legend_scale is None: 
            legend_scale = [np.min(peak_heights), np.max(peak_heights)]
        plt_color = np_scale_colors(legend_scale[0], legend_scale[1], peak_heights)

    if type(plt_color) is np.ndarray:
        p = sns.jointplot(x = x_ind, y = y_ind, marginal_kws={"bins": np.linspace(1, tair10.chr_inds[-1], 250),"color": hist_color }, xlim = (1, tair10.chr_inds[-1]), ylim = (1, tair10.chr_inds[-1]), kind = "scatter", alpha = 0.1, joint_kws={"s": 8})
        p.ax_joint.cla()
        #for i in range(len(x_ind)):
        p.ax_joint.scatter(x = x_ind, y = y_ind, c=plt_color, s = 8)
    else:
        p = sns.jointplot(x = x_ind, y = y_ind, marginal_kws={"bins": np.linspace(1, tair10.chr_inds[-1], 250),"color": hist_color }, xlim = (1, tair10.chr_inds[-1]), ylim = (1, tair10.chr_inds[-1]), kind = "scatter", color = plt_color, alpha = 0.1, joint_kws={"s": 8})
    p.ax_marg_y.remove()
    p.ax_joint.plot( (0, 0) , (1,tair10.chr_inds[-1]), '-', color = "gray")
    p.ax_joint.plot( (1,tair10.chr_inds[-1]), (0, 0), '-', color = "gray")
    for i in range(len(tair10.chrs)):
        p.ax_joint.plot( (tair10.chr_inds[i + 1] , tair10.chr_inds[i + 1]) , (1,tair10.chr_inds[-1]), '-', color = "gray")
        p.ax_joint.plot( (1,tair10.chr_inds[-1]), (tair10.chr_inds[i + 1] , tair10.chr_inds[i + 1]), '-', color = "gray")
        p.ax_joint.plot( (1,tair10.chr_inds[-1]), (tair10.chr_inds[i] + tair10.centro_mid[i] , tair10.chr_inds[i] + tair10.centro_mid[i]), ':k', color = "gray")
        p.ax_joint.plot( (tair10.chr_inds[i] + tair10.centro_mid[i] , tair10.chr_inds[i] + tair10.centro_mid[i]), (1,tair10.chr_inds[-1]), ':k', color = "gray")
    p.ax_joint.set_xticks( tair10.chr_inds[0:5] + (np.array(tair10.golden_chrlen)/2) )
    p.ax_joint.set_xticklabels( tair10.chrs )
    p.ax_joint.set_yticks( tair10.chr_inds[0:5] + (np.array(tair10.golden_chrlen)/2) )
    p.ax_joint.set_yticklabels( tair10.chrs )
    if legend_scale is not None:
        p.ax_marg_x.remove()
        norm = plt.Normalize(legend_scale[0], legend_scale[1])
        sm = plt.cm.ScalarMappable(cmap="BuGn", norm=norm)
        sm.set_array([])
        p.ax_joint.figure.colorbar(sm, shrink = 0.5)
    p.ax_joint.set_xlabel( "SNPs" )
    p.ax_joint.set_ylabel( "genes" )
    return(p)


def generate_manhattanplot(x_ind, y_ind, tair10, axs = None, plt_color=None, ylim = None, thres=None, gap = 10000000, s = 6, line = False, **kwargs):
    """
    Function to plot manhattan
        Required arguments: 
            x_ind : genome-wide indices from tair10 get_genomewide_inds function
            y_ind : Y-axis values to plot
            tair10: Class function from pyGenome package

        Plotting arguments:
            plt_color : HEX colors for all 5 chromosomes, else default colors are taken
            ylim: y-axis limit for plot
            thres: plot a line for threshold if given
            gap: can add in the gap between chromosomes
            line: True/False to determine if the plot should be line or scatter
    """
    if plt_color is None:
        plt_color = tair10.def_color
    if type(plt_color) is str or len(plt_color) < len(tair10.chrs):
        plt_color = [plt_color] * len(tair10.chrs)
    if ylim is None:
        if np.isnan(np.nanmax(y_ind)) or np.isinf(np.nanmax(y_ind)):
            ylim = [-0.1, 2]
        else:
            ylim = [-0.1, np.nanmax(y_ind) + 10]
    if axs is None:
        axs = plt.gca()
    # q = plt.scatter([0], [0], s = 0)
    for ix in range(len(tair10.chrs)):
        t_chr_ix = np.where((x_ind <= tair10.chr_inds[ix+1]) & ( x_ind > tair10.chr_inds[ix] ))[0]
        #y_ind[t_chr_ix[-1]] = 0
        #y_ind[t_chr_ix[0]] = 0
        if line:
            axs.plot(x_ind[t_chr_ix] + (gap * ix) , y_ind[t_chr_ix], color = plt_color[ix], **kwargs)
        else:
            axs.scatter(x_ind[t_chr_ix] + (gap * ix), y_ind[t_chr_ix], s = s, c = plt_color[ix], **kwargs)
        #q.axes.plot( [tair10.chr_inds[ix] + tair10.centro_mid[ix] + (gap * ix), tair10.chr_inds[ix] + tair10.centro_mid[ix] + (gap * ix)], ylim, '--', color = '#bdbdbd')
        #q.axes.plot( [tair10.chr_inds[ix+1] + (gap * ix), tair10.chr_inds[ix+1] + (gap * ix)], [0, ylim], 'k-', color = '#bdbdbd')
        # plt.xticks( np.arange(0, tair10.golden_chrlen[0], 10000000 ), [millify(ef) for ef in np.arange(0, tair10.golden_chrlen[0], 10000000 )] )
    axs.set_xticks( tair10.chr_inds[0:-1] + (np.arange(len(tair10.chrs)) * gap) + (np.array(tair10.golden_chrlen)/2) )
    axs.set_xticklabels( tair10.chrs )
    if thres is not None:
        axs.plot((0, tair10.chr_inds[-1] + (gap * (len(tair10.chrs) - 1))  ), (thres, thres), "--", color = "gray")
    axs.set_xlabel( "markers" )
    axs.set_ylim( ylim )
    axs.set_xlim( [0, tair10.chr_inds[-1] + (gap * (len(tair10.chrs) - 1)) ] )
    return(axs)

def generate_manhattanplot_bed(bed_df, genome_size_dict, plt_options = None, axs = None, **kwargs):
    """
    Function to plot manhattan
        Required arguments: 
        x_ind
        y_ind
            bed_df : pandas dataframe with (chr, pos, score)
            genome_size_dict = dictionary for chromosome IDs and its lengths
            x_ind

        Plotting arguments:
            plt_color : HEX colors for all 5 chromosomes, else default colors are taken
            ylim: y-axis limit for plot
            thres: plot a line for threshold if given
            gap: can add in the gap between chromosomes
            line: True/False to determine if the plot should be line or scatter
    """
    if plt_options is None:
        plt_options = {}
    genome_size_dict = pd.Series(genome_size_dict)
    if "color" not in plt_options.keys():
        plt_options['color'] = pd.Series( cb.diverging.PRGn_5.hex_colors[0], genome_size_dict.index )
        plt_options['color'][1::2] = cb.diverging.PRGn_5.hex_colors[4]
    if "pt_size" not in plt_options.keys():
        plt_options['pt_size'] = 6
    if "line" not in plt_options.keys():
        plt_options['line'] = False
    if "gap" not in plt_options.keys():
        plt_options['gap'] = 10000000
    if "ylim" not in plt_options.keys():
        plt_options['ylim'] = bed_df.iloc[:,2].describe().loc[["min", "max"]].to_list()

    if axs is None:
        axs = plt.gca()
    follow_x_index = 0
    chr_tick_position = []
    for ef_chr in genome_size_dict.iteritems():
        t_chr_ix = np.where(bed_df.iloc[:,0] == ef_chr[0])[0]
        if plt_options['line']:
            
            axs.plot(bed_df.iloc[t_chr_ix, 1].values + follow_x_index, bed_df.iloc[t_chr_ix,2], color = plt_options['color'][ef_chr[0]], **kwargs)
        else:
            axs.scatter(bed_df.iloc[t_chr_ix, 1].values + follow_x_index, bed_df.iloc[t_chr_ix,2], c = plt_options['color'][ef_chr[0]], s = plt_options['pt_size'], **kwargs)
        chr_tick_position.append( follow_x_index + (ef_chr[1]/2) )
        follow_x_index = follow_x_index + plt_options['gap'] + ef_chr[1]  ### adding gap and chromosome size
        #q.axes.plot( [tair10.chr_inds[ix] + tair10.centro_mid[ix] + (gap * ix), tair10.chr_inds[ix] + tair10.centro_mid[ix] + (gap * ix)], ylim, '--', color = '#bdbdbd')
        #q.axes.plot( [tair10.chr_inds[ix+1] + (gap * ix), tair10.chr_inds[ix+1] + (gap * ix)], [0, ylim], 'k-', color = '#bdbdbd')
        # plt.xticks( np.arange(0, tair10.golden_chrlen[0], 10000000 ), [millify(ef) for ef in np.arange(0, tair10.golden_chrlen[0], 10000000 )] )
    axs.set_xticks( chr_tick_position )
    axs.set_xticklabels( genome_size_dict.index )
    if "thres" in plt_options.keys():
        axs.plot((0, follow_x_index ), (plt_options["thres"], plt_options["thres"]), "--", color = "gray")
    axs.set_ylim( plt_options['ylim'] )
    return(axs)




def generate_heatmap_genomewide(score_df, tair10, plt_options = {}, axs = None, **kwargs):
    """
    Function to generate a genome wide heatmap
        plots chromosomes separately
        boxes for pericentromers

    """
    assert isinstance(score_df, pd.DataFrame), "provide a dataframe object"

    if "lims" not in plt_options.keys():
        plt_options['lims'] = (np.nanmin(score_df.values), np.nanmax(score_df.values))
    if "xlabel" not in plt_options.keys():
        plt_options['xlabel'] = ""
    if "ylabel" not in plt_options.keys():
        plt_options['ylabel'] = ""
    if "title" not in plt_options.keys():
        plt_options['title'] = ""
    if "cmap" not in plt_options.keys():
        plt_options['cmap'] = "Reds"
    if "col_centro_line" not in plt_options.keys():
        plt_options['col_centro_line'] = "#636363"
    if "col_chr_line" not in plt_options.keys():
        plt_options['col_chr_line'] = "#525252"

    if axs is None:
        axs = plt.gca()
    
    score_df_new = score_df.copy()
    if pd.api.types.is_string_dtype( score_df_new.columns ):
        score_df_new.columns = tair10.get_genomewide_inds( pd.Series(score_df_new.columns) )
    else:
        score_df_new.columns = score_df_new.columns.astype(int)
    
    chr_end_ix = np.searchsorted( score_df_new.columns.values, tair10.get_genomewide_inds( pd.DataFrame({"chr":tair10.chrs, "pos": tair10.golden_chrlen}) ), "right" )[0:-1]
    chr_centro_mid_ix = np.searchsorted( score_df_new.columns.values, tair10.get_genomewide_inds( pd.DataFrame({"chr":tair10.chrs, "pos": tair10.centro_mid}) ), "right" )
    chr_mid_ix = np.searchsorted( score_df_new.columns.values, tair10.get_genomewide_inds( pd.DataFrame({"chr":tair10.chrs, "pos": (np.array(tair10.golden_chrlen)/2).astype(int) }) ), "right" )
    # score_df_new.insert(loc = int(e_chr_end_ix), column = "break" + str(e_chr_end_ix), value = np.nan)
    
    sns.heatmap(
        score_df_new,
        vmin=plt_options['lims'][0], 
        vmax=plt_options['lims'][1], 
        cmap=plt_options['cmap'],
        ax = axs,
        xticklabels=False,
        yticklabels=False,
        **kwargs
        # cbar_kws=dict(use_gridspec=False,location="bottom")
    )
    axs.set_xlabel(plt_options['xlabel'])
    axs.set_ylabel(plt_options['ylabel'])
    axs.set_title(plt_options['title'])
    
    for e_chr_end_ix in chr_end_ix:
        axs.plot( (e_chr_end_ix, e_chr_end_ix), (0,score_df_new.shape[0]), '-', c = plt_options['col_chr_line'] )
        
    for e_chr_pos_ix in chr_centro_mid_ix:
        axs.plot( (e_chr_pos_ix, e_chr_pos_ix), (0,score_df_new.shape[0]), '--', c = plt_options['col_centro_line'] )
    axs.set_xticks( chr_mid_ix )
    axs.set_xticklabels( tair10.chrs )


## Make a PCA plot for SNPs using allel (scikit) package pca output
def generate_pca_from_snps(snps, sample_population=None, title="", pop_colors=None, plot_pca3=False):
    ## Provide sample_population to color dots
    import allel
    if sample_population is not None:
        ## Check if same number of accessions are provided
        assert sample_population.shape[0] == snps.shape[1]
    coords, model = allel.randomized_pca(snps, scaler=None )
    if plot_pca3:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        plot_pca_coords(coords, model, 0, 1, ax, sample_population, pop_colors)
        ax = fig.add_subplot(1, 2, 2)
        plot_pca_coords(coords, model, 2, 3, ax, sample_population, pop_colors)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        fig.suptitle(title, y=1.05)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        plot_pca_coords(coords, model, 0, 1, ax, sample_population, pop_colors)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        fig.suptitle(title, y=1.05)
        fig.tight_layout()

def plot_pca_coords(coords, model, pc1, pc2, ax, sample_population, pop_colors=None):
    sns.despine(ax=ax, offset=5)
    x = coords[:, pc1]
    y = coords[:, pc2]
    populations = np.unique(sample_population)
    if pop_colors == None:
        pop_colors = np.repeat("#7fcdbb", len(populations))
    for idx, pop in enumerate(populations):
        flt = (sample_population == pop)
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', color=pop_colors[idx],
                label=pop, markersize=6, mec='k', mew=.5)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))

def ternaryPlot(
        data,
        # Scale data for ternary plot (i.e. a + b + c = 1)
        scaling=True,
        # Direction of first vertex.
        start_angle=90,
        # Orient labels perpendicular to vertices.
        rotate_labels=True,
        # Labels for vertices.
        labels=('one','two','three'),
        # Can accomodate more than 3 dimensions if desired.
        sides=3,
        # Offset for label from vertex (percent of distance from origin).
        label_offset=0.10,
        # Any matplotlib keyword args for plots.
        edge_args={'color':'black','linewidth':2},
        # Any matplotlib keyword args for figures.
        fig_args = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},
        ):
    '''
    This will create a basic "ternary" plot (or quaternary, etc.)
    '''
    basis = array(
                    [
                        [
                            cos(2*_*pi/sides + start_angle*pi/180),
                            sin(2*_*pi/sides + start_angle*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = dot(data,basis)

    fig = figure(**fig_args)
    ax = fig.add_subplot(111)

    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotate_labels:
            angle = 180*arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + label_offset),
                y*(1 + label_offset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)

    # Plot border
    ax.plot(
        [basis[_,0] for _ in range(sides) + [0,]],
        [basis[_,1] for _ in range(sides) + [0,]],
        **edge_args
    )

    return newdata,ax


