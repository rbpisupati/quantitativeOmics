import numpy as np
import scipy as sp
import h5py as h5
import pandas as pd
import dask.dataframe as ddf
import os.path
import palettable.colorbrewer as cb
import numba as nb

from scipy.signal import find_peaks
import multiprocessing as mp
from . import parsers

def write_h5_qtls(input_file, tair10, marker_names, output_file, models_used = ['g', 'cis', 'cim', 'cis_given', 'gpluse', 'gxe', 'pval_nocis']):
    print( "reading in the file" )
    qtls_df = pd.read_csv(input_file, header=None, index_col=0)
    import ipdb; ipdb.set_trace()
    print("done")
    gene_list = qtls_df.index[qtls_df.index.str.endswith('.' + models_used[0])].str.replace( '.' + models_used[0], "" ).values
    gene_list = gene_list[np.argsort(tair10.get_genomewide_inds( pd.Series(gene_list) ))]
    print("writing a h5 file")
    with  h5.File(output_file, 'w') as h5file:
        h5file.create_dataset('genes', data=np.array(gene_list).astype("S") )
        h5file.create_dataset('markers', data=np.array(marker_names).astype("S") )
        h5file.create_dataset('marker_pos', data=np.array(tair10.get_genomewide_inds( marker_names )) )
        for ef_model in models_used:
            ef_peaks_df = qtls_df.loc[qtls_df.index.str.endswith('.' + ef_model),]
            if ef_peaks_df.shape[0] > 0:
                ef_peaks_df.index = pd.Series(ef_peaks_df.index.str.replace( '.' + ef_model, "" )).values
                ef_peaks_df = ef_peaks_df.reindex( gene_list )
                print("model %s, shape %s" % (ef_model, ef_peaks_df.shape))
                h5file.create_dataset(ef_model, data=ef_peaks_df.values, chunks=(1000, marker_names.shape[0] ))
    print("finished!")
    
    
class readQTLresults(parsers.readRqtlMaps):

    def __init__(self, input_genetic_map, qtl_results_h5_file, genome_class, marker_id_startswith = "Chr", marker_id_split = ":", marker_names = ['AA', 'AB', 'BB']):
        super().__init__(input_maps = {"map": input_genetic_map}, marker_id_startswith = marker_id_startswith, marker_id_split = marker_id_split, marker_names = marker_names  )
        self.genome_class = genome_class
        self.markers_map['genome_ix'] = self.genome_class.get_genomewide_inds( self.markers_map ) 
        # _,inType = os.path.splitext(qtl_results_h5_file)
        # if inType == '.hdf5':
        self.h5file = h5.File(qtl_results_h5_file, 'r')
        assert np.array(self.h5file['markers']).shape[0] == self.markers_map.shape[0], "Markers in genetic map do not match with results"
        t_gene_str = np.array(self.h5file['genes']).astype('U')
        self.gene_str = pd.Series( t_gene_str ).str.split( ",", expand = True )
        self.gene_str.index = t_gene_str
        self.gene_str['genome_ix'] = self.genome_class.get_genomewide_inds(pd.Series(t_gene_str))
        # self.csvfile = h5.File(qtl_results_file, 'r')
        # self.gene_str.columns = ['chr', 'start', 'end']

    def filter_gene_ix(self, req_gene_str):
        filter_ix = np.where( np.in1d(self.gene_str.index, req_gene_str) )[0]
        return(filter_ix)
    
    def get_matrix(self, model, req_gene_str):
        ef_gene_ix = self.filter_gene_ix( req_gene_str )
        ef_qtls = np.array(self.h5file[model][ef_gene_ix,:])
        #  self.gene_str['genome_ix'][ef_gene_ix].values
        return( pd.DataFrame( ef_qtls, columns= self.markers_map['genome_ix'].values, index = req_gene_str ) )
    
    def get_long_matrix(self, model, req_gene_str, qtl_min = 3, qtl_max = 15):
        ef_matrix = self.get_matrix( model, req_gene_str )
        ef_matrix[ef_matrix < qtl_min] = np.nan
        ef_matrix[ef_matrix >= qtl_max] = qtl_max
        ###__________________________-
        ef_matrix.loc[:,'gene_ix'] = self.genome_class.get_genomewide_inds( pd.Series(ef_matrix.index) )
        ef_matrix = pd.melt( ef_matrix, id_vars=['gene_ix'], value_vars= ef_matrix.columns[ef_matrix.columns != "gene_ix"].values, var_name = "marker_ix", value_name="LOD" ).dropna()
        ef_matrix['marker_ix'] = ef_matrix['marker_ix'].astype(int)
        ef_matrix = ef_matrix.sample(frac=1)
        ef_matrix['peak_dist'] = np.abs(ef_matrix['gene_ix'] - ef_matrix['marker_ix'])
        return( ef_matrix )
    
    def get_peaks(self, model, req_gene_str, nprocs = 8, qtl_min = 3, qtl_max = None):
        ## get the peaks for each gene . using scipy
        ## https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        genes_lods_df = self.get_matrix(model, req_gene_str)
        output_peaks = determine_peaks_from_wide_matrix_of_lods(genes_lods_df, nprocs = nprocs, qtl_min = qtl_min, qtl_max = qtl_max, n_smooth = 200 )
        output_peaks['genome_gene_ind'] = self.gene_str['genome_ix'].loc[ output_peaks['gene_str'] ].values 
        return(output_peaks)
        
    def calculate_gxe( self, g_null = 'g', g_add = 'gpluse', g_int = 'gxe' ):
        null_model = self.get_matrix( g_null, self.gene_str )
        add_model = self.get_matrix( g_add, self.gene_str )
        add_model = add_model.reindex( null_model.index )
        gxe_model = self.get_matrix( g_int, self.gene_str )
        gxe_model = gxe_model.reindex( null_model.index )
        gxe_model = gxe_model - add_model
        add_model = add_model - null_model
        return( { "add": add_model, "int": gxe_model } )

def determine_peaks_from_wide_matrix_of_lods(genes_lods_df, nprocs = 8, qtl_min = 3, qtl_max = None, n_smooth = 250):
    """
    Function to get the peaks from a wide matrix
    rows: genes
    cols: markers 
    with lod scores
    """
    genes_lods_ddf = ddf.from_pandas(genes_lods_df, npartitions=nprocs)   # where the number of partitions is the number of cores you want to use
    output_peaks = genes_lods_ddf.apply( lambda x: determine_peaks_given_lods(x, qtl_min = qtl_min, n_smooth = n_smooth), meta=('int'), axis=1 ).compute(scheduler='multiprocessing')
    if output_peaks.apply(len).sum() > 0:
        output_peaks = output_peaks.apply(pd.Series).stack().reset_index(drop=False)
        output_peaks.columns = np.array( ["gene_str", "qtl_num", "marker_ix"] )
        output_peaks['marker_ix'] = output_peaks['marker_ix'].astype(int)
        output_peaks['genome_marker_ind'] = genes_lods_df.columns.values[ output_peaks['marker_ix'].astype(int) ]
        output_peaks['qtl_num'] = output_peaks['qtl_num'].astype(int) + 1
        output_peaks['lod'] = np.array([genes_lods_df.at[ef[0], ef[1]] for ef in zip(output_peaks['gene_str'].values, output_peaks['genome_marker_ind'] )])
        if qtl_max is not None and str(qtl_max).isnumeric():
            output_peaks.loc[output_peaks['lod'] > float(qtl_max), 'lod'] = float(qtl_max)
        return(output_peaks)
    else:
        return( pd.DataFrame( columns = ["gene_str", "qtl_num", "marker_ix", "lod"] ) )


def determine_peaks_given_lods(lod_scores, qtl_min = 3, peak_prominence = 1, n_smooth = 250, **kwargs):
    ##  Using a smoothing function -- convolve
    ## Find peaks in scipy to determine the 
    smooth_lods = smooth_sum(lod_scores, n_smooth)
    smooth_lods = np.concatenate(([min(smooth_lods)],smooth_lods,[min(smooth_lods)]))
    gene_peaks = find_peaks(smooth_lods, height = qtl_min, prominence = peak_prominence, **kwargs)
    return(gene_peaks[0]-1)
    # gene_peaks = np.array([], dtype = int)
    # for ef_chr in self.marker_indices.keys():
    #     ef_chr_gene_lods = smooth_sum(lod_scores[self.marker_indices[ef_chr]], n_smooth)
    #     ## Need to add a minimum at the first and last to detect a peak at the first position
    #     ef_chr_gene_lods = np.concatenate(([min(ef_chr_gene_lods)],ef_chr_gene_lods,[min(ef_chr_gene_lods)]))
    #     ef_chr_gene_peaks = find_peaks(ef_chr_gene_lods, height = qtl_min, prominence = peak_prominence, **kwargs)
    #     if len(ef_chr_gene_peaks[0]) > 0:
    #         gene_peaks = np.append(gene_peaks, self.marker_indices[ef_chr][ef_chr_gene_peaks[0]-1])
    # return(gene_peaks)
    

def smooth_sum(arr, n_times= 100):
    arr = np.array(arr, dtype = float)
    for e_ind in np.arange(n_times):
        arr = np.insert(arr, 0, arr[0])
        arr = np.append(arr, arr[-1] )
        arr = sp.signal.convolve( arr, [0.25,0.5,0.25], mode = "same" )[1:-1]
#         arr = sp.signal.convolve( arr, [0.25,0.5,0.25] )[1:-1]
    return(arr)



@nb.njit(parallel = True)
def genome_rotations(lod_scores, npermute = 100):
    ##  
    num_markers = lod_scores.shape[0]
    output_matrix = np.zeros( (npermute, num_markers) )
    for ef in nb.prange( npermute ):
        ef_rng = np.random.choice(np.arange(num_markers))
        output_matrix[ef,:] = np.concatenate((lod_scores[ef_rng:], lod_scores[0:ef_rng]))
    return(output_matrix)


@nb.njit(parallel = True)
def genome_rotations_2d(lod_scores_2d, npermute = 100):
    ##  
    num_markers = lod_scores_2d.shape[1]
    num_lines = lod_scores_2d.shape[0]
    output_matrix = np.zeros( (npermute, num_markers) )
    for ef in nb.prange( npermute ):
        t_perm_out = np.zeros( (num_lines, num_markers) )
        for ef_line in nb.prange( num_lines ):
            ef_rng = np.random.choice(np.arange(num_markers))
            t_perm_out[ef_line,:] = np.concatenate((lod_scores_2d[ef_line, ef_rng:], lod_scores_2d[ef_line, 0:ef_rng]))
        output_matrix[ef,:] = np.sum(t_perm_out, axis = 0 )
    return(output_matrix)


def invLogit(x):
    return(1 / (1 + np.exp(-x)))

def logit(x):
    return(np.log((x/(1 - x))))

def transform_beta(x, s = 0.5):
    return( (x * (x.shape[0] - 1 ) + s) / len(x) )

def perform_tukey_hsd(endog, groups, groups_order, alpha = 0.01):
    groups_mod = utils.marker_to_int( groups, groups_order).astype(str)
    ef_model_tukey = pairwise_tukeyhsd(endog = endog, groups = groups_mod, alpha = alpha)
    ef_model_tukey = pd.DataFrame(data=ef_model_tukey._results_table.data[1:], columns=ef_model_tukey._results_table.data[0])
    ef_model_tukey = ef_model_tukey.set_index(ef_model_tukey['group1'] + "_" + ef_model_tukey['group2'] )
    return(ef_model_tukey)


def linear_model_tukey(ef_data_phenos):
    phenos_list = ['CG', 'CHG', 'CHH']
    marker_list = ['AA', 'AB', 'BB']
    temp_list = ['T4', "T16"]
    cisxtemp_list = ['AA_T4', 'AA_T16', 'AB_T4', 'AB_T16', 'BB_T4', 'BB_T16']
    
    ef_model_data = pd.Series(dtype = float)
    ef_model_data = ef_model_data.append(pd.Series((ef_data_phenos['cisxtemp'] == cisxtemp_list[1]).sum(), index = ['counts_AA_T16']  ) )
    ef_model_data = ef_model_data.append(pd.Series((ef_data_phenos['cisxtemp'] == cisxtemp_list[5]).sum(), index = ['counts_BB_T16']  ) )
    for ef_pheno in phenos_list:
        ef_model_data = ef_model_data.append(pd.Series(ef_data_phenos[ef_pheno].describe()[["mean", 'std']].values,      index = ['meths_mean.' + ef_pheno, 'meths_std.' + ef_pheno]  ) )

        ef_model = anova.anova_lm( smapi.ols( ef_pheno + " ~ temp + marker ", data = ef_data_phenos).fit(), typ = 2 )
        ef_model = ef_model.reindex( ['temp', 'marker', 'Residual'] )
        ef_model_data = ef_model_data.append(pd.Series(ef_model.loc["temp", "sum_sq"],      index = ['ss_temp.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_model.loc["marker", "sum_sq"],    index = ['ss_cis.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_model.loc["Residual", "sum_sq"],  index = ['ss_residual.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_model.loc["temp", "PR(>F)"],      index = ['pval_temp.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_model.loc["marker", "PR(>F)"],    index = ['pval_cis.' + ef_pheno]  ) )
        
        ef_tukey_temp = perform_tukey_hsd(ef_data_phenos[ef_pheno], ef_data_phenos['temp'], groups_order = temp_list)
        ef_tukey_cis = perform_tukey_hsd(ef_data_phenos[ef_pheno], ef_data_phenos['marker'], groups_order = marker_list)
        ef_tukey_cistemp = perform_tukey_hsd(ef_data_phenos[ef_pheno], ef_data_phenos['cisxtemp'], groups_order = cisxtemp_list)
        
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_temp.reindex(["1_2"]).loc["1_2", "meandiff"],   index = ['eff_temp_diff.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_temp.reindex(["1_2"]).loc["1_2", "lower"],      index = ['eff_temp_lwr.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_temp.reindex(["1_2"]).loc["1_2", "upper"],      index = ['eff_temp_upr.' + ef_pheno]  ) )
        
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cis.reindex(["1_3"]).loc["1_3", "meandiff"],    index = ['eff_AA.BB_diff.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cis.reindex(["1_3"]).loc["1_3", "lower"],       index = ['eff_AA.BB_lwr.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cis.reindex(["1_3"]).loc["1_3", "upper"],       index = ['eff_AA.BB_upr.' + ef_pheno]  ) )
        
        ef_model_data = ef_model_data.append(pd.Series(ef_data_phenos[ef_data_phenos['cisxtemp'] == cisxtemp_list[1] ][ef_pheno].describe()[['mean', 'std']].values,    index = ['meths_AA_T16.mean.' + ef_pheno, 'meths_AA_T16.std.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["1_2"]).loc["1_2", "meandiff"],    index = ['eff_temp.AA_diff.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["1_2"]).loc["1_2", "lower"],       index = ['eff_temp.AA_lwr.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["1_2"]).loc["1_2", "upper"],       index = ['eff_temp.AA_upr.' + ef_pheno]  ) )

        ef_model_data = ef_model_data.append(pd.Series(ef_data_phenos[ef_data_phenos['cisxtemp'] == cisxtemp_list[5] ][ef_pheno].describe()[['mean', 'std']].values,    index = ['meths_BB_T16.mean.' + ef_pheno, 'meths_BB_T16.std.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["5_6"]).loc["5_6", "meandiff"],    index = ['eff_temp.BB_diff.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["5_6"]).loc["5_6", "lower"],       index = ['eff_temp.BB_lwr.' + ef_pheno]  ) )
        ef_model_data = ef_model_data.append(pd.Series(ef_tukey_cistemp.reindex(["5_6"]).loc["5_6", "upper"],       index = ['eff_temp.BB_upr.' + ef_pheno]  ) )
    
    return(ef_model_data)

