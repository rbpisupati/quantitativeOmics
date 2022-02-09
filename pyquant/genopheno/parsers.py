"""
Data parsers
"""
import numpy as np
import scipy as st
import pandas as pd
import h5py
import sys
import logging
import os.path
import csv
# from pygwas.core import genotype


log = logging.getLogger(__name__)
def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def readPhenoData(phenoFile, geno):
    # use pandas to read the file
    log.info("loading phenotype file")
    sniffer = csv.Sniffer()
    tpheno = open(phenoFile, 'rb')
    ifheader = sniffer.has_header(tpheno.read(4096))
    tpheno.seek(0)
    sniff_pheno = sniffer.sniff(tpheno.read(4096))
    if ifheader:
        pheno = pd.read_table(phenoFile, header = 0, sep=sniff_pheno.delimiter)
    else:
        pheno = pd.read_table(phenoFile, header = None, sep=sniff_pheno.delimiter)
    pheno.columns = np.array(['acc','pheno'])
    reqPheno, reqAccsInd = getCommonPheno(pheno, geno)
    return(reqPheno, reqAccsInd)

def getCommonPheno(pheno, geno):
    # taking one temperature into account
    accs_1001g = np.array(geno.accessions)
    reqPheno = []
    reqAccsInd = []
    reqAccs = []
    pheno_accs = np.array(pheno['acc'], dtype="string")
    for i in range(len(accs_1001g)):
        ind = np.where(pheno_accs == accs_1001g[i])[0]
        if len(ind) > 1:
            reqPheno.append(np.mean(pheno['pheno'][ind]))
            reqAccsInd.append(i)
            reqAccs.append(accs_1001g[i])
        elif len(ind) == 1:
            reqPheno.append(pheno['pheno'][ind[0]])
            reqAccsInd.append(i)
            reqAccs.append(accs_1001g[i])
    reqPheno = np.array(reqPheno)
    if len(reqPheno) == 0:
        raise NotImplementedError
    return(reqPheno, reqAccsInd)

H5_EXT = ['.hdf5', '.h5', 'h5py']

def readGenotype(genoFile):
    fileName,fileType = os.path.splitext(genoFile)
    if fileType in H5_EXT:
        log.info("loading genotype file")
        g = genotype.load_hdf5_genotype_data(genoFile)
        return(g)
    else:
        raise NotImplementedError

def readGenotype_acc(genoFile):
    # only for ending in hdf5
    fileName,fileType = os.path.splitext(genoFile)
    if os.path.isfile(fileName + '.acc' + fileType):
        g_acc = genotype.load_hdf5_genotype_data(fileName + '.acc' + fileType)
        return(g_acc)
    else:
        return(None)

def readKinship(kinFile, reqAccsInd):
    log.info("loading kinship file")
    kinship1001g = h5py.File(kinFile)
    return(kinship1001g['kinship'][reqAccsInd,:][:,reqAccsInd])

class InputsfurLimix(object):

    def __init__(self, genoFile, kinFile, phenoFile= None, pheno_type=None, transform=None, test=None):
        self.geno = readGenotype(genoFile)
        self.pheno_type = pheno_type
        self.test = test
        self.pheno = self.parse_pheno(phenoFile, transform)
        self.parseKinFile(kinFile)

    def parse_pheno(self, phenoFile, transform):
        if phenoFile is None:
            self.accinds = range(len(self.geno.accessions))
            return(None)
        from pygwas.core import phenotype
        reqPheno, self.accinds = readPhenoData(phenoFile, self.geno)
        pheno = phenotype.Phenotype(self.geno.accessions[self.accinds], reqPheno, None)
        if transform is not None:
            pheno.transform(transform)
        return(pheno)

    def parseKinFile(self, kinFile):
        if kinFile is not None:
            self.kin = readKinship(kinFile, self.accinds)
        else:
            from . import kinship
            self.kin = kinship.calc_kinship(self.geno)



### A temporary function that need to be removed while publishing
def split_ids(sample_ids):
    sample_ids_df = pd.Series(sample_ids, index = sample_ids).str.split("_", expand = True)
    sample_ids_df[['plate', 'position']] =  sample_ids_df.iloc[:,1].str.split(".", expand = True)
    sample_ids_df['dir_temp_plate'] = sample_ids_df.iloc[:,3] + "_" + sample_ids_df.iloc[:,0] + "_" + sample_ids_df['plate']
    sample_ids_df['genotype'] = sample_ids_df.iloc[:,2] + "_" + sample_ids_df.iloc[:,3]
    sample_ids_df['dir_temp'] = sample_ids_df.iloc[:,3] + "_" + sample_ids_df.iloc[:,0]
    sample_ids_df['dir_plate'] = sample_ids_df.iloc[:,3] + "_" + sample_ids_df['plate']
    sample_ids_df['temp_plate'] = sample_ids_df.iloc[:,0] + "_" +  sample_ids_df['plate']
    sample_ids_df.index = sample_ids
    sample_ids_df.rename(columns={0: 'temp', 1: 'plate_id', 2: 'genotype', 3: 'direction'}, inplace=True)
    return(sample_ids_df)



class readRqtlMaps(object):

    def __init__(self, input_maps, marker_id_startswith = "chr", marker_id_split = ":", marker_names = ['AA', 'AB', 'BB']):
        self.maps = []
        self._marker_names = marker_names
        self.sample_ids = pd.DataFrame()
        for each_map_id in input_maps.keys():
            self.load_rqtl_map(each_map_id, input_maps[each_map_id], marker_id_startswith = marker_id_startswith, marker_id_split=marker_id_split)
            self.maps.append(each_map_id)
            self.sample_ids = pd.concat([ self.sample_ids, self.__getattribute__("ids_" + each_map_id)  ])

    def load_rqtl_map(self, map_key, input_map, marker_id_startswith = "Chr", marker_id_split = ":"):
        qtl_map = pd.read_csv(input_map, index_col=0, header=None)
        marker_ix = pd.Series(qtl_map.index).str.contains(marker_id_startswith, case = False)
        if marker_id_split is not None:
            qtl_markers = qtl_map.index[marker_ix].astype(str)
            qtl_markers_df = pd.Series(qtl_markers).str.split(marker_id_split, expand = True)
            qtl_markers_df.columns = ['chr', 'start']
            qtl_markers_df['start'] = qtl_markers_df['start'].astype(int)
            qtl_markers_df['end'] = qtl_markers_df['start'] + 1
            qtl_markers_df.index = qtl_markers
        else:
            qtl_markers_df = pd.DataFrame(index = qtl_map.index[marker_ix].astype(str) )
            qtl_markers_df['chr'] = qtl_map.loc[qtl_map.index[marker_ix],:].iloc[:,0]
            qtl_markers_df['pos'] = qtl_map.loc[qtl_map.index[marker_ix],:].iloc[:,1]
        id_preset = qtl_map.index.str.contains("id", case = False)
        if id_preset.sum() > 0:
            sample_ids_df = split_ids(qtl_map.iloc[np.where(id_preset)[0][0],2:].values) #### always values are coming from third column
        else:
            sample_ids_df = pd.DataFrame( index = "id_" + map_key + "_" + pd.Series(np.arange(qtl_map.shape[1] - 2), dtype = 'str') )
        qtl_map.columns = np.append(np.array(["chr", "pos"]), sample_ids_df.index)
        setattr(self, "map_" + map_key, qtl_map)
        setattr(self, 'markers_' + map_key, qtl_markers_df)
        setattr(self, "ids_" + map_key, sample_ids_df)

    def get_genotype_marker(self, map_key, marker_id, return_int = False):
        if type(marker_id) is int:
            marker_to_return = self.__getattribute__("map_" + map_key).loc[self.__getattribute__( "markers_" + map_key ).index[marker_id], self.__getattribute__( "ids_" + map_key ).index ]
        else:
            marker_to_return = self.__getattribute__("map_" + map_key).loc[marker_id, self.__getattribute__( "ids_" + map_key ).index ]
        if return_int:
            return( utils.marker_to_int(marker_to_return, marker_ids = self._marker_names) )
        return(marker_to_return)

    def get_closest_marker(self, map_key, bed_str ):
        import pybedtools as pybed
        marker_bed = self.__getattribute__( "markers_" + map_key ).iloc[:,[0,1,2]].copy()
        marker_bed["marker_id"] = marker_bed.index.values
        query_bed =  pd.Series( bed_str ).str.split(",", expand = True).iloc[:,[0,1,2]].apply(pd.to_numeric, errors='ignore')
        query_bed['index'] = query_bed.index
        query_bed = pybed.BedTool.from_dataframe( query_bed )
        marker_bed = pybed.BedTool.from_dataframe( marker_bed.sort_values( ['chr', 'start'] )  )
        closest_markers = query_bed.closest( marker_bed, k = 1 ).to_dataframe()
        closest_markers.columns = np.array(['query_chr', 'query_start', 'query_end', 'query_index', 'marker_chr', 'marker_start', 'marker_end', 'marker_id'])
        return( closest_markers )

    def geno_pheno(self, map_key, marker_id, phenos_df ):
        geno_df =  self.get_genotype_marker(map_key, marker_id)
        if type(geno_df) is pd.Series:
            # geno_df = pd.DataFrame({"marker0": geno_df})
            geno_df = pd.DataFrame( geno_df )
        elif type(geno_df) is pd.DataFrame:
            geno_df = geno_df.T
            # geno_df.columns = 'marker' + pd.Series(np.arange(geno_df.shape[1]), dtype = str)
        if type(phenos_df) is pd.Series:
            phenos_df = pd.DataFrame( phenos_df )
        return( pd.merge(geno_df, phenos_df, right_index = True, left_index = True ) )
        
    def write_h5_qtl_scores(self, map_key, input_qtl_lods, output_file, model_sep_on_rows = ".", csv_sep = "\t"):
        # models_used = ['g', 'cis', 'cim', 'cis_given', 'gpluse', 'gxe', 'pval_nocis']
        print( "reading in the file" )
        marker_names = self.__getattribute__( "markers_" + map_key ).index.values
        qtls_df = pd.read_csv(input_qtl_lods, sep = csv_sep, header=None, index_col=0)
        print("done")
        assert marker_names.shape[0] == qtls_df.shape[1], "columns should be markers, their shapes do not match"
        # import ipdb; ipdb.set_trace()
        # if model_sep_on_rows is not None:
        all_qtls_given = pd.Series(qtls_df.index).str.split(model_sep_on_rows, expand = True)
        models_used = all_qtls_given.iloc[:,1].unique()
        gene_list = all_qtls_given.iloc[:,0].unique()
        chunks_sas = min(1000, gene_list.shape[0])
        # gene_list = qtls_df.index[qtls_df.index.str.endswith('.' + models_used[0])].str.replace( '.' + models_used[0], "" ).values
        # gene_list = gene_list[np.argsort(tair10.get_genomewide_inds( pd.Series(gene_list) ))]
        print("writing a h5 file")
        with  h5py.File(output_file, 'w') as h5file:
            h5file.create_dataset('genes', data=np.array(gene_list).astype("S") )
            h5file.create_dataset('markers', data=np.array(self.__getattribute__( "markers_" + map_key )).astype("S") )
            # h5file.create_dataset('marker_pos', data=np.array(tair10.get_genomewide_inds( marker_names )) )
            for ef_model in models_used:
                ef_peaks_df = qtls_df.loc[qtls_df.index.str.endswith('.' + ef_model),]
                if ef_peaks_df.shape[0] > 0:
                    ef_peaks_df.index = pd.Series(ef_peaks_df.index.str.replace( '.' + ef_model, "" )).values
                    ef_peaks_df = ef_peaks_df.reindex( gene_list )
                    print("model %s, shape %s" % (ef_model, ef_peaks_df.shape))
                    h5file.create_dataset(ef_model, data=ef_peaks_df.values, chunks=(chunks_sas, marker_names.shape[0] ))
        print("finished!")
    
        