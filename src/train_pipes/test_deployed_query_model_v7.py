
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.22.1',
        'google-api-core==2.10.0',
        'google-cloud-pipeline-components'
    ],
)
def test_deployed_query_model(
    project: str,
    location: str,
    version: str,
    deployed_endpoint: str,
    data_dir_bucket_name: str,
    test_instance_gcs_blob_name: str,
    # instances: list,
    metrics: Output[Metrics],
):
    # here
    import base64
    import logging

    from google.cloud import aiplatform
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import \
        GcpResources
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    import pickle as pkl
    import time

    logging.getLogger().setLevel(logging.INFO)
    aiplatform.init(project=project)
    storage_client = storage.Client(project=project)

    # parse endpoint resource
    logging.info(f"Endpoint = {deployed_endpoint}")
    gcp_resources = Parse(deployed_endpoint, GcpResources())
    endpoint_uri = gcp_resources.resources[0].resource_uri
    endpoint_id = "/".join(endpoint_uri.split("/")[-8:-2])
    logging.info(f"Endpoint ID = {endpoint_id}")

    # define endpoint client
    _endpoint = aiplatform.Endpoint(endpoint_id)
    
    # ====================================================
    # Load test instance
    # ====================================================
    LOCAL_INSTANCE_FILE = 'merlin_last5_test_instance.pkl'
    logging.info(f"LOCAL_INSTANCE_FILE: {LOCAL_INSTANCE_FILE}")
    
    bucket = storage_client.bucket(data_dir_bucket_name)
    blob = bucket.blob(test_instance_gcs_blob_name)
    blob.download_to_filename(LOCAL_INSTANCE_FILE)

    filehandler = open(LOCAL_INSTANCE_FILE, 'rb')
    test_instances_dict = pkl.load(filehandler)
    filehandler.close()
    
    logging.info(f'test_instances_dict: {test_instances_dict}')
    
    # ====================================================
    # prediction request
    # ====================================================

    start = time.time()

    playlist_emb = _endpoint.predict(instances=[test_instances_dict])
    
    end = time.time()
    
    elapsed_time = end - start
    elapsed_time = round(elapsed_time, 2)
    logging.info(f'Deployed query model latency: {elapsed_time} seconds')
    logging.info(f'query embeddings: {playlist_emb.predictions}')
    
    metrics.log_metric("endpoint latency", elapsed_time)
