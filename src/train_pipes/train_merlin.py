
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.18.1',
        # 'google-cloud-storage',
    ],
)
def train_merlin(
    project: str,
    location: str,
    version: str,
    train_image_uri: str,     # TODO: Artifact
    tb_resource: str,
    batch_size: int, 
    train_epochs: int,
    train_dir: str,
    valid_dir: str,
    workflow_dir: str,
    experiment_name: str,
    experiment_run: str,
    service_account: str,
) -> NamedTuple('Outputs', [
    ('merlin_model_gcs_dir', str),
    ('query_tower_gcs_dir', str),
    ('candidate_tower_gcs_uri', str),
    ('candidate_embeddings_gcs_uri', str),
]):
    
    import logging
    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime
    import time

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    # ====================================================
    # Helper function for workerpool specs
    # ====================================================
    def prepare_worker_pool_specs(
        image_uri,
        # args,
        cmd,
        replica_count=1,
        machine_type="n1-standard-16",
        accelerator_count=1,
        accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
        reduction_server_count=0,
        reduction_server_machine_type="n1-highcpu-16",
        reduction_server_image_uri="us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest",
    ):

        if accelerator_count > 0:
            machine_spec = {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            }
        else:
            machine_spec = {"machine_type": machine_type}

        container_spec = {
            "image_uri": image_uri,
            # "args": args,
            "command": cmd,
        }

        chief_spec = {
            "replica_count": 1,
            "machine_spec": machine_spec,
            "container_spec": container_spec,
        }

        worker_pool_specs = [chief_spec]
        if replica_count > 1:
            workers_spec = {
                "replica_count": replica_count - 1,
                "machine_spec": machine_spec,
                "container_spec": container_spec,
            }
            worker_pool_specs.append(workers_spec)
        if reduction_server_count > 1:
            workers_spec = {
                "replica_count": reduction_server_count,
                "machine_spec": {
                    "machine_type": reduction_server_machine_type,
                },
                "container_spec": {"image_uri": reduction_server_image_uri},
            }
            worker_pool_specs.append(workers_spec)

        return worker_pool_specs
    
    # ====================================================
    # Define device strategy
    # ====================================================
    # TODO: parameterize
    
    WORKER_MACHINE_TYPE = 'a2-highgpu-1g'
    REPLICA_COUNT = 1
    ACCELERATOR_TYPE = 'NVIDIA_TESLA_A100'
    PER_MACHINE_ACCELERATOR_COUNT = 1
    REDUCTION_SERVER_COUNT = 0                                                      
    REDUCTION_SERVER_MACHINE_TYPE = "n1-highcpu-16"
    DISTRIBUTE_STRATEGY = 'single'
    
    # ====================================================
    # # DEFINE ARGS
    # ====================================================
    # TODO: parameterize
    
    BATCH_SIZE = 4096*4      # TODO: `batch_size * 4 ? jw
    LEARNING_RATE = 0.001
    LAYERS = "[512, 256, 128]"
    
    OUTPUT_BUCKET = 'jt-merlin-scaling'
    
    EXPERIMENT_RUN = f'{experiment_run}-{TIMESTAMP}'
    
    WORKER_CMD = [
        'sh',
        '-euc',
        f'''pip freeze && python -m trainer.train_task --tb_name={tb_resource} --per_gpu_batch_size={batch_size} \
        --train_output_bucket={OUTPUT_BUCKET} --train_dir={train_dir} --valid_dir={valid_dir} --workflow_dir={workflow_dir} \
        --num_epochs={train_epochs} --learning_rate={LEARNING_RATE} --distribute={DISTRIBUTE_STRATEGY} \
        --experiment_name={experiment_name} --experiment_run={EXPERIMENT_RUN} --project={project} --location={location}'''
    ]
    
    WORKER_POOL_SPECS = prepare_worker_pool_specs(
        image_uri=train_image_uri,
        # args=WORKER_ARGS,
        cmd=WORKER_CMD,
        replica_count=REPLICA_COUNT,
        machine_type=WORKER_MACHINE_TYPE,
        accelerator_count=PER_MACHINE_ACCELERATOR_COUNT,
        accelerator_type=ACCELERATOR_TYPE,
        reduction_server_count=REDUCTION_SERVER_COUNT,
        reduction_server_machine_type=REDUCTION_SERVER_MACHINE_TYPE,
    )
    # ==============================================================================
    # Submit Train Job 
    # ==============================================================================
    STAGING_BUCKET = f'gs://{OUTPUT_BUCKET}/{experiment_name}'
    JOB_NAME = f'train-merlin-retrieval-{version}'
    gpu_type = ACCELERATOR_TYPE.lower() # lowercase for labels

    job = vertex_ai.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=WORKER_POOL_SPECS,
        staging_bucket=STAGING_BUCKET,
        labels={
            'gpu': f'{gpu_type}',
            'gpu_per_replica' : f'{PER_MACHINE_ACCELERATOR_COUNT}',
            'replica_cnt' : f'{REPLICA_COUNT}',
        }
    )
    
    job.run(
        sync=True, 
        service_account=service_account,
        # tensorboard=EXPERIMENT_TB,
        restart_job_on_worker_restart=False,
        enable_web_access=True,
    )
    
    # uris set during train script
    WORKING_DIR_GCS_URI = f'gs://{OUTPUT_BUCKET}/{experiment_name}/{EXPERIMENT_RUN}'
    MODEL_DIR = f"{WORKING_DIR_GCS_URI}/model-dir"
    QUERY_TOWER_PATH = f"{MODEL_DIR}/query-tower"
    CANDIDATE_TOWER_PATH = f"{MODEL_DIR}/candidate-tower"
    EMBEDDINGS_PATH = f"{MODEL_DIR}/candidate-embeddings"
    
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
    )
