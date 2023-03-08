
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.22.1',
        'google-cloud-pipeline-components',
        'google-cloud-storage',
        'numpy'
    ],
)
def test_model_index_endpoint(
    project: str,
    location: str,
    version: str,
    data_dir_bucket_name: str,
    test_instance_gcs_blob_name: str,
    ann_index_endpoint_resource_uri: str,
    brute_index_endpoint_resource_uri: str,
    endpoint: str, # Input[Artifact],
    metrics: Output[Metrics],
    # metrics: Output[Metrics],
    # metrics: Output[Metrics],
    # metrics: Output[Metrics],
):
    import logging
    import time
    import numpy as np
    import pickle as pkl
    
    import base64

    from typing import Dict, List, Union

    from google.cloud import aiplatform as vertex_ai
    from google.protobuf.json_format import Parse
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import \
        GcpResources
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    # import tensorflow as tf

    logging.getLogger().setLevel(logging.INFO)

    vertex_ai.init(
        project=project,
        location=location,
    )
    storage_client = storage.Client(project=project)
    
    # ====================================================
    # get deployed model endpoint
    # ====================================================
    logging.info(f"Endpoint = {endpoint}")
    gcp_resources = Parse(endpoint, GcpResources())
    logging.info(f"gcp_resources = {gcp_resources}")
    
    _endpoint_resource = gcp_resources.resources[0].resource_uri
    logging.info(f"_endpoint_resource = {_endpoint_resource}")
    
    _endpoint_uri = "/".join(_endpoint_resource.split("/")[-8:-2])
    logging.info(f"_endpoint_uri = {_endpoint_uri}")
    
    # define endpoint resource in component
    _endpoint = vertex_ai.Endpoint(_endpoint_uri)
    logging.info(f"_endpoint defined")
    
    # ====================================================
    # Get indexes
    # ====================================================
    logging.info(f"ann_index_endpoint_resource_uri: {ann_index_endpoint_resource_uri}")
    logging.info(f"brute_index_endpoint_resource_uri: {brute_index_endpoint_resource_uri}")

    deployed_ann_index = vertex_ai.MatchingEngineIndexEndpoint(ann_index_endpoint_resource_uri)
    deployed_bf_index = vertex_ai.MatchingEngineIndexEndpoint(brute_index_endpoint_resource_uri)

    DEPLOYED_ANN_ID = deployed_ann_index.deployed_indexes[0].id
    DEPLOYED_BF_ID = deployed_bf_index.deployed_indexes[0].id
    logging.info(f"DEPLOYED_ANN_ID: {DEPLOYED_ANN_ID}")
    logging.info(f"DEPLOYED_BF_ID: {DEPLOYED_BF_ID}")
    
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
    # get query response
    # ====================================================
    start = time.time()

    playlist_emb = _endpoint.predict(instances=[test_instances_dict])
    
    end = time.time()
    
    elapsed_query_time = end - start
    elapsed_query_time = round(elapsed_query_time, 8)
    logging.info(f'Query endpoint latency: {elapsed_query_time} seconds')
    
    # ====================================================
    # call matching engine with predicted emb vectors
    # ====================================================
    logging.info('Retreiving neighbors from ANN index...')
    start = time.time()
    ANN_response = deployed_ann_index.match(
        deployed_index_id=DEPLOYED_ANN_ID,
        queries=playlist_emb.predictions,
        num_neighbors=50
    )
    end = time.time()
    elapsed_ann_time = end - start
    elapsed_ann_time = round(elapsed_ann_time, 8)
    logging.info(f'ANN latency: {elapsed_ann_time} seconds')
    
    
    logging.info('Retreiving neighbors from BF index...')
    start = time.time()
    BF_response = deployed_bf_index.match(
        deployed_index_id=DEPLOYED_BF_ID,
        queries=playlist_emb.predictions,
        num_neighbors=50
    )
    end = time.time()
    elapsed_bf_time = end - start
    elapsed_bf_time = round(elapsed_bf_time, 8)
    logging.info(f'Bruteforce latency: {elapsed_bf_time} seconds')
    
    # TODO: write results to file -> GCS
    
    # ====================================================
    # Calculate recall by determining how many neighbors were correctly retrieved 
    # compare with brute-force search
    # ====================================================
    recalled_neighbors = 0
    for tree_ah_neighbors, brute_force_neighbors in zip(
        ANN_response, BF_response
    ):
        tree_ah_neighbor_ids = [neighbor.id for neighbor in tree_ah_neighbors]
        brute_force_neighbor_ids = [neighbor.id for neighbor in brute_force_neighbors]

        recalled_neighbors += len(
            set(tree_ah_neighbor_ids).intersection(brute_force_neighbor_ids)
        )

    recall = recalled_neighbors / len(
        [neighbor for neighbors in BF_response for neighbor in neighbors]
    )

    logging.info("Recall: {}".format(recall))
    logging.info(f'playlist_emb: {playlist_emb.predictions}')
    logging.info(f'ANN_response: {ANN_response}')
    logging.info(f'BF_response: {BF_response}')
    
    metrics.log_metric("elapsed_query_time", elapsed_query_time)
    metrics.log_metric("elapsed_ann_time", elapsed_ann_time)
    metrics.log_metric("elapsed_bf_time", elapsed_bf_time)
    metrics.log_metric("Recall", (recall * 100.0))
