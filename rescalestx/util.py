import scanpy as sc
import pandas as pd
import os
import pdb
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import anndata


def read_10x_object(metadata, raw=False):
    # metadata is a pandas row
    folder_name = os.path.dirname(metadata['samples'])
    if raw == True:
        count_file = metadata['samples_raw']
    else:
        count_file = metadata['samples']
    adata = sc.read_10x_h5(filename = count_file)
    fields = metadata.index.tolist()
    fields.remove('samples')
    for field in fields:
        adata.obs[field] = [metadata[field]]*adata.n_obs
        adata.obs[field] = adata.obs[field].astype('category')

    adata.var['symbol'] = adata.var_names
    adata.var_names = adata.var['gene_ids']
    #pdb.set_trace()
    #adata.var['symbol'] = adata.var['symbol'].where(~adata.var['symbol'].duplicated(), adata.var['symbol'].index + '_1')
    adata.obs_names=[f"{x}-{metadata['sample_names']}" for x in adata.obs_names]
    adata.raw = adata
    return(adata)

def read_10x_object_table(infoTable):
    h5_objs = []
    for ind,row in infoTable.iterrows():
        adata = read_10x_object(row)
        h5_objs.append(adata)
    return(h5_objs)

def switch2symbol(h5_obj):
    if all(h5_obj.var_names == h5_obj.var['gene_ids']):
        h5_obj.var['gene_ids'] = h5_obj.var_names
        h5_obj.var_names = h5_obj.var['symbol']
    else:
        print('h5 already symbol')
    return(h5_obj)
        
def switch2ensembl(h5_obj):
    if all(h5_obj.var_names == h5_obj.var['symbol']):
        h5_obj.var['symbol'] = h5_obj.var_names
        h5_obj.var_names = h5_obj.var['gene_ids']
    else:
        print('h5 already ensembl')
    return(h5_obj)

def merge_h5s(h5_list):
    all_h5s = h5_list[0].copy()

    for i in range(1,len(h5_list)):
        all_h5s=anndata.concat([all_h5s,h5_list[i]],uns_merge="unique",join="inner",merge="same")
    return(all_h5s)