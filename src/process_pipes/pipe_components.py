"""KFP components."""

from typing import Optional
from . import config

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output

# =============================================
#            convert_to_parquet_op
# =============================================
@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def convert_parquet_op(
    output_dataset: Output[Dataset],
    bucket_name: str,
    data_prefix: str,
    file_pattern: str,
    output_path_defined_dir: str,
    # data_dir_pattern: str,
    # data_paths: list,
    split: str,
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
    '''
    Component to create NVTabular definition.
    
    Args:
    output_dataset: Output[Dataset]
      Output metadata with references to the converted CSV files in GCS
      and the split name.The path to the files are in GCS fuse format:
      /gcs/<bucket name>/path/to/file
    bucket: gcs bucket holding train & valid data
    data_path_prefix: file path to GCS blobl object (e.g., gs://...data/path/prefix.../blob.xxx)
    data_paths: list
    split: str
      Split name of the dataset. Example: train or valid
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    memory_limit: Optional[int] = 100_000_000_000
    '''
    
    # =========================================================
    #            import packages
    # =========================================================
    import os
    import logging
    from google.cloud import storage
    
    storage_client = storage.Client()

    from preprocess_task import (
        create_cluster,
        create_parquet_dataset_definition,
        convert_definition_to_parquet,
        # get_criteo_col_dtypes,
    )
    
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    logging.info('Base path in %s', output_dataset.path)
    # =========================================================
    #            Define data paths
    # =========================================================
    logging.info(f'bucket_name: {bucket_name}')
    logging.info(f'data_prefix: {data_prefix}')
    
    # Write metadata
    output_dataset.metadata['split'] = split

    logging.info('Creating cluster')
    create_cluster(
        n_workers=n_workers,
        device_limit_frac=device_limit_frac,
        device_pool_frac=device_pool_frac,
        memory_limit=memory_limit
    )
    
    # logging.info(f'Creating dataset definition from: {data_path_prefix}')
    dataset = create_parquet_dataset_definition(
        bucket_name=bucket_name,
        data_prefix=data_prefix,
        frac_size=frac_size,
        file_pattern=file_pattern,
    )
    
    logging.info(f'Converting Definition to Parquet; {output_dataset.uri}')
    logging.info(f'Parquet Definition Output Path: ; {output_path_defined_dir}/{split}')
    convert_definition_to_parquet(
        output_path=f'{output_path_defined_dir}/{split}', # output_dataset.uri,
        dataset=dataset,
        output_files=num_output_files,
        shuffle=shuffle
    )
    
# =========================================================
#            analyze_dataset_op
# =========================================================
@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    output_path_defined_dir: str,
    output_path_analyzed_dir: str,
    n_workers: int,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
    '''
    Component to generate statistics from the dataset.
    
    Args:
    parquet_dataset: List of strings
      Input metadata with references to the train and valid converted
      datasets in GCS and the split name.
    workflow: Output[Artifact]
      Output metadata with the path to the fitted workflow artifacts
      (statistics).
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    '''
    import logging
    import nvtabular as nvt
  
    from preprocess_task import (
        create_cluster,
        create_nvt_workflow,
    )

    logging.basicConfig(level=logging.INFO)

    create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac,
      memory_limit=memory_limit
    )
    
    # logging.info(f'Creating Parquet dataset:{parquet_dataset.uri}')
    logging.info(f'Creating Parquet dataset output_path_defined_dir: {output_path_defined_dir}/train')
    dataset = nvt.Dataset(
        path_or_source=f'{output_path_defined_dir}/train', # TODO: JT Check "train"    # parquet_dataset.uri,
        engine='parquet',
        part_mem_fraction=frac_size,
        suffix='.parquet'
    )

    logging.info('Creating Workflow')
    # Create Workflow
    nvt_workflow = create_nvt_workflow()

    logging.info('Analyzing dataset')
    nvt_workflow = nvt_workflow.fit(dataset)

    logging.info('Saving Workflow')
    nvt_workflow.save(f'{output_path_analyzed_dir}') # workflow.path)
    
# =========================================================
#            transform_dataset_op
# =========================================================
@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    output_path_defined_dir: str,
    output_path_transformed_dir: str,
    output_path_analyzed_dir: str,
    version: str,
    bucket_data_src: str,
    bucket_data_output: str,
    app: str,
    split: str,
    num_output_files: int,
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.6,
    device_pool_frac: float = 0.9,
    frac_size: float = 0.10,
    memory_limit: int = 100_000_000_000
):
    """Component to transform a dataset according to the workflow definitions.
    Args:
        workflow: Input[Artifact]
        Input metadata with the path to the fitted_workflow
        parquet_dataset: Input[Dataset]
              Location of the converted dataset in GCS and split name
        transformed_dataset: Output[Dataset]
        Split name of the transformed dataset.
        shuffle: str
            How to shuffle the converted CSV, default to None. Options:
                PER_PARTITION
                PER_WORKER
                FULL
    device_limit_frac: float = 0.6
    device_pool_frac: float = 0.9
    frac_size: float = 0.10
    """
    
    import os
    import logging
    import nvtabular as nvt
    from merlin.schema import Tags

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    from preprocess_task import (
        create_cluster,
        save_dataset,
    )
    def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
        """Uploads a file to GCS bucket"""
        client = storage.Client()
        blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
        blob.bucket._client = client
        blob.upload_from_filename(source_file_name)
    
    def _read_blob_gcs(bucket_name, source_blob_name, destination_filename):
        """Downloads a file from GCS to local directory"""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_filename)
        

    logging.basicConfig(level=logging.INFO)

    transformed_dataset.metadata['split'] = split
    
    logging.info('Creating cluster')
    create_cluster(
        n_workers=n_workers,
        device_limit_frac=device_limit_frac,
        device_pool_frac=device_pool_frac,
        memory_limit=memory_limit
    )

   # logging.info(f'Creating Parquet dataset:gs://{parquet_dataset.uri}')
    logging.info(f'Creating Parquet dataset:{output_path_defined_dir}/{split}')
    dataset = nvt.Dataset(
        path_or_source=f'{output_path_defined_dir}/{split}', #f'gs://{parquet_dataset.uri}',
        engine='parquet',
        part_mem_fraction=frac_size,
        suffix='.parquet'
    )
    
    logging.info('Loading Workflow')
    nvt_workflow = nvt.Workflow.load(f'{output_path_analyzed_dir}') # workflow.path)

    logging.info('Transforming Dataset')
    trans_dataset = nvt_workflow.transform(dataset)

    logging.info(f'transformed_dataset.uri: {transformed_dataset.uri}')
    logging.info(f'Saving transformed dataset: {output_path_transformed_dir}/{split}')
    save_dataset(
        dataset=trans_dataset,
        output_path=f'{output_path_transformed_dir}/{split}', # transformed_dataset.uri,
        output_files=num_output_files,
        shuffle=shuffle
    )
    logging.info(f'transformed_dataset saved!')
    logging.info(f'transformed_dataset.path: {transformed_dataset.path}')
    
    # =========================================================
    #        read and upload files
    # =========================================================
    '''
    nv-tabular creates a txt file with all `gs://` paths
    create a copy that replaces `gs://` with `/gcs/`
    '''
    logging.info('Generating file list for training...')
    
#     # get loca directory
#     # LOCAL_DIRECTORY = os.getcwd()
#     LOCAL_DIRECTORY = '/tmp/directory'
    
#     # _bucket_name='spotify-merlin-v1' # bucket_data_src
#     PREFIX = f'nvt-preprocessing-{app}-{version}/nvt-processed/{split}'
#     FILENAME = '_file_list.txt'
#     SOURCE_BLOB_NAME = f'{PREFIX}/{FILENAME}'
#     logging.info(f'SOURCE_BLOB_NAME: {SOURCE_BLOB_NAME}')
    
#     # LOCAL_DESTINATION_FILENAME = f'{LOCAL_DIRECTORY}/local_file_list.txt'
#     LOCAL_DESTINATION_FILENAME = 'local_file_list.txt'
#     logging.info(f'LOCAL_DESTINATION_FILENAME: {LOCAL_DESTINATION_FILENAME}')
    
#     _read_blob_gcs(
#         bucket_name=bucket_data_output,
#         source_blob_name=f'{SOURCE_BLOB_NAME}', 
#         destination_filename=LOCAL_DESTINATION_FILENAME
#     )
    
#     file_list = os.path.join(transformed_dataset.path, '_file_list.txt')
    
#     # write new '/gcs/' file
#     new_lines = []
#     with open(LOCAL_DESTINATION_FILENAME, 'r') as fp:
#         lines = fp.readlines()
#         new_lines.append(lines[0])
#         for line in lines[1:]:
#             new_lines.append(line.replace('gs://', '/gcs/'))

#     NEW_LOCAL_FILENAME = f'{LOCAL_DIRECTORY}/_gcs_file_list.txt'
#     logging.info(f'NEW_LOCAL_FILENAME: {NEW_LOCAL_FILENAME}')
    
#     with open(NEW_LOCAL_FILENAME, 'w') as fp:
#         fp.writelines(new_lines)
        
#     GCS_URI_DESTINATION = f'{output_path_transformed_dir}/{split}'
#     logging.info(f'GCS_URI_DESTINATION: {GCS_URI_DESTINATION}')
    
#     _upload_blob_gcs(
#         gcs_uri=GCS_URI_DESTINATION, 
#         source_file_name=NEW_LOCAL_FILENAME, 
#         destination_blob_name='_gcs_file_list.txt'
#     )
# logging.info(f'List of /gcs/ file paths uploaded to {GCS_URI_DESTINATION}/_gcs_file_list.txt')

#     file_list_name = '_file_list.txt'
#     file_list_uri = f'{output_path_transformed_dir}/{split}/{file_list_name}'
#     logging.info(f'file_list_uri : {file_list_uri}')

#     new_lines = []
#     with open(file_list_uri, 'r') as fp:
#         lines = fp.readlines()
#         new_lines.append(lines[0])
#         for line in lines[1:]:
#             new_lines.append(line.replace('gs://', '/gcs/'))

#     gcs_file_list_name = '_gcs_file_list.txt'
#     gcs_file_list_uri = f'{output_path_transformed_dir}/{split}/{gcs_file_list_name}'
#     logging.info(f'gcs_file_list_uri : {gcs_file_list_uri}')
    
#     with open(gcs_file_list_uri, 'w') as fp:
#         fp.writelines(new_lines)
    
#     logging.info(f'List of /gcs/ file paths uploaded to {gcs_file_list}')
    
    # =========================================================
    #        Saving cardinalities
    # =========================================================
    logging.info('Saving cardinalities')
    
    cols_schemas = nvt_workflow.output_schema.select_by_tag(Tags.CATEGORICAL)
    cols_names = cols_schemas.column_names

    cards = []
    for c in cols_names:
        col = cols_schemas.get(c)
        cards.append(col.properties['embedding_sizes']['cardinality'])

    transformed_dataset.metadata['cardinalities'] = cards
    # transformed_dataset.metadata['dataset_gcs_uri'] = gcs_file_list
