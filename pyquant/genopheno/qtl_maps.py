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


def write_h5_qtls(input_file, tair10, marker_names, output_file, models_used = ['g', 'cis', 'cim', 'cis_given', 'gpluse', 'gxe', 'pval_nocis']):
    print( "reading in the file" )
    qtls_df = pd.read_csv(input_file, header=None, index_col=0)
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
    
    
class readQTLdata(object):

    def __init__(self, hdf5_file, genome_class):
        self.genome_class = genome_class
        self.h5file = h5.File(hdf5_file,'r')
        self.markers = np.array(self.h5file['markers']).astype('U')
        self.markers = pd.DataFrame(self.markers, columns=["chr", "pos" ])
        self.markers['pos'] = self.markers['pos'].astype(int)
        self.markers['genome_ix'] = np.array(self.h5file['marker_pos']).astype("U").astype(int)
        self.marker_indices = self.markers.groupby(["chr"]).indices
        self.gene_str = np.array(self.h5file['genes']).astype('U')
        self.gene_genome_ind = self.genome_class.get_genomewide_inds(pd.Series(self.gene_str))
    
    def filter_gene_ix(self, req_gene_str):
        filter_ix = np.where( np.in1d(self.gene_str, req_gene_str) )[0]
        return(filter_ix)
    
    def get_matrix(self, model, req_gene_str):
        ef_gene_ix = self.filter_gene_ix( req_gene_str )
        ef_qtls = np.array(self.h5file[model][ef_gene_ix,:])
        return( pd.DataFrame( ef_qtls, columns= self.markers['genome_ix'].values, index = self.gene_str[ef_gene_ix] ) )
    
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
        genes_lods_ddf = ddf.from_pandas(genes_lods_df, npartitions=nprocs)   # where the number of partitions is the number of cores you want to use
        output_peaks = genes_lods_ddf.apply( lambda x: determine_peaks_given_lods(x, qtl_min = qtl_min ), meta=('int'), axis=1 ).compute(scheduler='multiprocessing')
        output_peaks = output_peaks.apply(pd.Series).stack().reset_index(drop=False)
        output_peaks.columns = np.array( ["gene_str", "qtl_num", "marker_ix"] )
        output_peaks['marker_ix'] = output_peaks['marker_ix'].astype(int)
        output_peaks['genome_marker_ind'] = self.markers['genome_ix'].values[output_peaks['marker_ix'].astype(int)]
        output_peaks['qtl_num'] = output_peaks['qtl_num'].astype(int) + 1
        output_peaks['genome_gene_ind'] = (pd.Series(self.gene_genome_ind, index = self.gene_str).loc[ output_peaks['gene_str'].values ]).values
        output_peaks['lod'] = np.array([genes_lods_df.at[ef[0], ef[1]] for ef in zip(output_peaks['gene_str'].values, output_peaks['genome_marker_ind'] )])
        if qtl_max is not None and str(qtl_max).isnumeric():
            output_peaks.loc[output_peaks['lod'] > float(qtl_max), 'lod'] = float(qtl_max)
        return(output_peaks)
        # for ef_gene_ix in req_gene_ix:
        #     ef_gene_peaks = self.determine_peaks_given_gene_ix(ef_gene_ix, model = model, n_smooth = n_smooth, qtl_min = qtl_min, peak_prominence = peak_prominence)
        #     out_peaks = out_peaks.append( pd.DataFrame( {"gene_ix": self.gene_genome_ind[ef_gene_ix], "marker_ix": self.markers['genome_ix'][ef_gene_peaks].values, "LOD": self.h5file[model][ef_gene_ix,:][ef_gene_peaks] } ), ignore_index = True)
        # mp_pool = mp.Pool( 12 )
        # mp_drones=[mp_pool.apply_async(self.determine_peaks_given_gene_ix, args=gene_ix) for gene_ix in req_gene_ix ] 
        # for drone in mp_drones: 
        #     Results.collectData(drone.get())    
        # mp_pool.close()
        # mp_pool.join() 
        # return( out_peaks )
        
    def calculate_gxe( self, g_null = 'g', g_add = 'gpluse', g_int = 'gxe' ):
        null_model = self.get_matrix( g_null, self.gene_str )
        add_model = self.get_matrix( g_add, self.gene_str )
        add_model = add_model.reindex( null_model.index )
        gxe_model = self.get_matrix( g_int, self.gene_str )
        gxe_model = gxe_model.reindex( null_model.index )
        gxe_model = gxe_model - add_model
        add_model = add_model - null_model
        return( { "add": add_model, "int": gxe_model } )

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