
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.23.0',
        # 'google-cloud-storage',
    ],
)
def train_merlin(
    project: str,
    location: str,
    version: str,
    train_image_uri: str,     # TODO: Artifact
    train_output_gcs_bucket: str,
    tb_resource: str,
    batch_size: int, 
    train_epochs: int,
    train_dir: str,
    valid_dir: str,
    workflow_dir: str,
    experiment_name: str,
    experiment_run: str,
    service_account: str,
    worker_pool_specs: dict,
) -> NamedTuple('Outputs', [
    ('merlin_model_gcs_dir', str),
    ('query_tower_gcs_dir', str),
    ('candidate_tower_gcs_uri', str),
    ('candidate_embeddings_gcs_uri', str),
    ('working_dir_gcs_path', str),
]):
    
    import logging
    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime
    import time
    
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    
    BASE_OUTPUT_DIR = f'gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}'
    STAGING_BUCKET = f'{BASE_OUTPUT_DIR}/staging'
    
    logging.info(f'BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}')
    logging.info(f'STAGING_BUCKET: {STAGING_BUCKET}')

    vertex_ai.init(
        project=project,
        location=location,
        experiment=experiment_name,
    )
    
    # ====================================================
    # # DEFINE ARGS
    # ====================================================
    # TODO: parameterize
    # worker_pool_specs[0]['container_spec']['command'].append(f'--tb_name={tb_resource}')
    JOB_NAME = f'mm-2t-pipe-train-{version}'
    
    logging.info(f'tensorboard_resource_name: {tb_resource}')
    logging.info(f'service_account: {service_account}')
    logging.info(f'worker_pool_specs: {worker_pool_specs}')
    logging.info(f'JOB_NAME: {JOB_NAME}')
    # logging.info(f'gpu_type: {gpu_type}')
    # ==============================================================================
    # Submit Train Job 
    # ==============================================================================

    job = vertex_ai.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=BASE_OUTPUT_DIR,
        staging_bucket=STAGING_BUCKET,
    )
    
    job.run(
        sync=False, 
        service_account=service_account,
        tensorboard=tb_resource,
        restart_job_on_worker_restart=False,
        enable_web_access=True,
    )
    
    # uris set during train script
    WORKING_DIR_GCS_URI = f'gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}'
    MODEL_DIR = f"{WORKING_DIR_GCS_URI}/model_dir"
    QUERY_TOWER_PATH = f"{MODEL_DIR}/query_tower"
    CANDIDATE_TOWER_PATH = f"{MODEL_DIR}/candidate_tower"
    EMBEDDINGS_PATH = f"{MODEL_DIR}/candidate_embeddings"
    
    logging.info(f'WORKING_DIR_GCS_URI: {WORKING_DIR_GCS_URI}')
    logging.info(f'MODEL_DIR: {MODEL_DIR}')
    logging.info(f'QUERY_TOWER_PATH: {QUERY_TOWER_PATH}')
    logging.info(f'CANDIDATE_TOWER_PATH: {CANDIDATE_TOWER_PATH}')
    logging.info(f'EMBEDDINGS_PATH: {EMBEDDINGS_PATH}')
    
    return (
        f'{MODEL_DIR}',
        f'{QUERY_TOWER_PATH}',
        f'{CANDIDATE_TOWER_PATH}',
        f'{EMBEDDINGS_PATH}',
        f'{WORKING_DIR_GCS_URI}',
    )
