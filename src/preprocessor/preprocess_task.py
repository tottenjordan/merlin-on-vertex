
import argparse
import logging
# logging.disable(logging.WARNING)
import os
import sys
import time
import numpy as np
from typing import Dict, List, Union

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import fsspec

import nvtabular as nvt
from merlin.io.shuffle import Shuffle
from nvtabular.ops import (
    Categorify,
    TagAsUserID,
    TagAsItemID,
    TagAsItemFeatures,
    TagAsUserFeatures,
    AddMetadata,
    ListSlice
)
import nvtabular.ops as ops
from nvtabular.utils import device_mem_size

from merlin.schema.tags import Tags
# import merlin.models.tf as mm
from merlin.io.dataset import Dataset

import tensorflow as tf

# for running this example on CPU, comment out the line below
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# =============================================
# featutres
# =============================================
item_id = ["track_uri_can"] >> Categorify(dtype="int32") >> TagAsItemID() 
playlist_id = ["pid"] >> Categorify(dtype="int32") >> TagAsUserID() 


item_features_cat = [
    'artist_name_can',
    'track_name_can',
    'artist_genres_can',
]

item_features_cont = [
    'duration_ms_can',
    'track_pop_can',
    'artist_pop_can',
    'artist_followers_can',
]

playlist_features_cat = [
    'description_pl',
    'name',
    'collaborative',
]

playlist_features_cont = [
    'duration_ms_seed_pl',
    'n_songs_pl',
    'num_artists_pl',
    'num_albums_pl',
]

seq_feats_cat = [
    'artist_name_pl',
    'track_uri_pl',
    'track_name_pl',
    'album_name_pl',
    'artist_genres_pl',
]

CAT = playlist_features_cat + item_features_cat
CONT = item_features_cont + playlist_features_cont

# item_feature_cat_node = item_features_cat >> nvt.ops.FillMissing()>> Categorify(dtype="int32") >> TagAsItemFeatures()

# item_feature_cont_node =  item_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsItemFeatures()

# playlist_feature_cat_node = playlist_features_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsUserFeatures() 

# playlist_feature_cont_node = playlist_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsUserFeatures()

# playlist_feature_cat_seq_node = seq_feats_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsUserFeatures() 

# =============================================
# create cluster
# =============================================
def create_cluster(
    n_workers,
    device_limit_frac,
    device_pool_frac,
    memory_limit
):
    """Create a Dask cluster to apply the transformations steps to the Dataset."""
    device_size = device_mem_size()
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    rmm_pool_size = (device_pool_size // 256) * 256

    cluster = LocalCUDACluster(
        n_workers=n_workers,
        device_memory_limit=device_limit,
        rmm_pool_size=rmm_pool_size,
        memory_limit=memory_limit
    )

    return Client(cluster)

# =============================================
#            Create & Save dataset
# =============================================

def create_parquet_nvt_dataset(
    # data_path,
    frac_size,
    data_prefix,
    bucket_name,
    file_pattern,
):
    """Create a nvt.Dataset definition for the parquet files."""
    
    # BUCKET = 'gs://spotify-builtin-2t'
    # DATA_PATH = f"{BUCKET}/{data_prefix}/0000000000**.snappy.parquet"
    DATA_PATH = f"gs://{bucket_name}/{data_prefix}/{file_pattern}" #0000000000**.snappy.parquet"
    logging.info(f"DATA_PATH: {DATA_PATH}")
    
    fs = fsspec.filesystem('gs')
    
    file_list = fs.glob(DATA_PATH)
        # os.path.join(data_path, '*.parquet')
    # )

    if not file_list:
        raise FileNotFoundError('Parquet file(s) not found')

    file_list = [os.path.join('gs://', i) for i in file_list]
    
    logging.info(f"Number of files: {len(file_list)}")

    # return nvt.Dataset(f"{bucket_name}/{data_prefix}/0000000000**.snappy.parquet", part_mem_fraction=frac_size)
    return nvt.Dataset(
        file_list,
        engine='parquet',
        part_mem_fraction=frac_size
  )

def save_dataset(
    dataset,
    output_path,
    output_files,
    # categorical_cols,
    # continuous_cols,
    shuffle=None,
):
    """Save dataset to parquet files to path."""
    categorical_cols=CAT
    continuous_cols=CONT

    dict_dtypes = {}
    for col in categorical_cols:
        dict_dtypes[col] = np.int32

    for col in continuous_cols:
        dict_dtypes[col] = np.float64

    dataset.to_parquet(
        output_path=output_path,
        shuffle=shuffle,
        output_files=output_files,
        dtypes=dict_dtypes,
        cats=categorical_cols,
        conts=continuous_cols,
    )

# =============================================
#            Workflow
# =============================================
def create_nvt_workflow():
    '''
    Create a nvt.Workflow definition with transformation all the steps
    '''
    item_id = ["track_uri_can"] >> Categorify(dtype="int32") >> TagAsItemID() 
    playlist_id = ["pid"] >> Categorify(dtype="int32") >> TagAsUserID() 


    item_features_cat = ['artist_name_can',
            'track_name_can',
            'artist_genres_can',
        ]

    item_features_cont = [
            'duration_ms_can',
            'track_pop_can',
            'artist_pop_can',
            'artist_followers_can',
        ]

    playlist_features_cat = [
            'description_pl',
            'name',
            'collaborative',
        ]

    playlist_features_cont = [
            'duration_ms_seed_pl',
            'n_songs_pl',
            'num_artists_pl',
            'num_albums_pl',
        ]

    seq_feats_cat = [
            'artist_name_pl',
            'track_uri_pl',
            'track_name_pl',
            'album_name_pl',
            'artist_genres_pl',
        ]

    CAT = playlist_features_cat + item_features_cat
    CONT = item_features_cont + playlist_features_cont

    item_feature_cat_node = item_features_cat >> nvt.ops.FillMissing()>> Categorify(dtype="int32") >> TagAsItemFeatures()

    item_feature_cont_node =  item_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsItemFeatures()

    playlist_feature_cat_node = playlist_features_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsUserFeatures() 

    playlist_feature_cont_node = playlist_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsUserFeatures()

    playlist_feature_cat_seq_node = seq_feats_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsUserFeatures()
    
    # define a workflow
    output = playlist_id + item_id \
    + item_feature_cat_node \
    + item_feature_cont_node \
    + playlist_feature_cat_node \
    + playlist_feature_cont_node \
    + playlist_feature_cat_seq_node 

    workflow = nvt.Workflow(output)
    
    return workflow

# =============================================
#            Create Parquet Dataset 
# =============================================

def create_parquet_dataset_definition(
    # data_paths,
    # recursive,
    # col_dtypes,
    frac_size,
    bucket_name,
    data_prefix,
    file_pattern,
    # sep='\t'
):
    from google.cloud import storage
    storage_client = storage.Client()
    
    DATASET_DEFINITION = f"gs://{bucket_name}/{data_prefix}/{file_pattern}"  # 0000000000**.snappy.parquet"
    
    logging.info(f'DATASET_DEFINITION: {DATASET_DEFINITION}')
    
    fs = fsspec.filesystem('gs')
    file_list = fs.glob(DATASET_DEFINITION)

    if not file_list:
        raise FileNotFoundError('Parquet file(s) not found')

    file_list = [os.path.join('gs://', i) for i in file_list]
    logging.info(f"Number of files: {len(file_list)}")
    
    return nvt.Dataset(f"{DATASET_DEFINITION}", engine='parquet', part_mem_fraction=frac_size)


def convert_definition_to_parquet(
    output_path,
    dataset,
    output_files,
    shuffle=None
):
    """Convert Parquet files to parquet and write to GCS."""
    if shuffle == 'None':
        shuffle = None
    else:
        try:
            shuffle = getattr(Shuffle, shuffle)
        except:
            print('Shuffle method not available. Using default.')
            shuffle = None

    dataset.to_parquet(
        output_path,
        shuffle=shuffle,
        output_files=output_files
    )
    
# =============================================
#            Create nv-tabular definition
# =============================================
def main_convert(args):
    
    logging.info('Beginning main-convert from preprocess_task.py...')
    logging.info(f'args.output_path: {args.output_path}')
    
    logging.info('Creating cluster')
    client = create_cluster(
        args.n_workers,
        args.device_limit_frac,
        args.device_pool_frac,
        args.memory_limit
    )
    
    logging.info('Creating parquet dataset definition')
    dataset = create_parquet_dataset_definition(
        # data_paths=args.parq_data_path,
        # recursive=False,
        bucket_name=args.bucket_name,     # 'spotify-builtin-2t', # TODO: parameterize
        data_prefix=args.data_prefix,     # 'train', # TODO: JT check
        frac_size=args.frac_size,
        file_pattern=file_pattern,
    )

    logging.info('Converting definition to Parquet')
    convert_definition_to_parquet(
        args.output_path,
        dataset,
        args.output_files
    )
    
# =============================================
#            Analyse Dataset 
# =============================================
def main_analyze(args):
    
    logging.info('Beginning main-analyze from preprocess_task.py...')
    logging.info(f'args.bucket_name: {args.bucket_name}')
    
    logging.info('Creating cluster')
    client = create_cluster(
        args.n_workers,
        args.device_limit_frac,
        args.device_pool_frac,
        args.memory_limit
    )
    
    logging.info('Creating Parquet dataset')
    dataset = create_parquet_nvt_dataset(
        # data_dir=args.parquet_data_path,
        frac_size=args.frac_size,
        data_prefix='train_data_parquet', # TODO: JT check
        bucket_name=args.bucket_name,
        file_pattern=file_pattern #"0000000000**.snappy.parquet",
    )
  
    logging.info('Creating Workflow')
    # Create Workflow
    nvt_workflow = create_nvt_workflow()
  
    logging.info('Analyzing dataset')
    nvt_workflow = nvt_workflow.fit(dataset)

    logging.info('Saving Workflow')
    nvt_workflow.save(args.output_path)
    
# =============================================
#            Transform Dataset 
# =============================================
def main_transform(args):
    
    logging.info('Beginning main-transform from preprocess_task.py...')
    logging.info(f'args.bucket_name: {args.bucket_name}')
    
    client = create_cluster(
        args.n_workers,
        args.device_limit_frac,
        args.device_pool_frac,
        args.memory_limit,
    )

    # nvt_workflow = create_nvt_workflow()
    nvt_workflow = nvt.Workflow.load(args.workflow_path, client)

    # dataset = create_parquet_nvt_dataset(
    #     args.parquet_data_path, 
    #     frac_size=args.frac_size)
    
    dataset = create_parquet_nvt_dataset(
        # data_dir=args.parquet_data_path,
        frac_size=args.frac_size,
        data_prefix='train_data_parquet', # TODO: JT check
        bucket_name=args.bucket_name,
        file_pattern=file_pattern #"0000000000**.snappy.parquet",
    )

    logging.info('Transforming Dataset')
    transformed_dataset = nvt_workflow.transform(dataset)

    logging.info('Saving transformed dataset')
    save_dataset(
        transformed_dataset,
        output_path=args.output_path,
        output_files=args.output_files,
        # categorical_cols=CAT,
        # continuous_cols=CONT,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
    )
    
# =============================================
#            args
# =============================================
def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
  
    parser.add_argument(
        '--task',
        type=str,
        required=False
    )
    parser.add_argument(
        '--bucket_name',
        type=str,
        required=False
    )
    parser.add_argument(
        '--parquet_data_path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--parq_data_path',
        required=False,
        nargs='+'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--output_files',
        type=int,
        required=False
    )
    parser.add_argument(
        '--workflow_path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        required=False
    )
    parser.add_argument(
        '--frac_size',
        type=float,
        required=False,
        default=0.10
    )
    parser.add_argument(
        '--memory_limit',
        type=int,
        required=False,
        default=100_000_000_000
    )
    parser.add_argument(
        '--device_limit_frac',
        type=float,
        required=False,
        default=0.60
    )
    parser.add_argument(
        '--device_pool_frac',
        type=float,
        required=False,
        default=0.90
    )

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )

    parsed_args = parse_args()

    start_time = time.time()
    logging.info('Timing task')

    if parsed_args.task == 'transform':
        main_transform(parsed_args)
    elif parsed_args.task == 'analyze':
        main_analyze(parsed_args)
    elif parsed_args.task == 'convert':
        main_convert(parsed_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info('Task completed. Elapsed time: %s', elapsed_time)
