
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.21.1',
        'google-api-core==2.10.0',
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

    logging.getLogger().setLevel(logging.INFO)
    aiplatform.init(project=project, staging_bucket=bucket)

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
    
    # # test instance #TODO: paramterize
    # TEST_INSTANCE = {
    #     'collaborative': 'false',
    #     'album_name_pl': [
    #         "There's Really A Wolf", 'Late Nights: The Album','American Teen', 'Crazy In Love', 'Pony'
    #     ], 
    #     'artist_genres_pl': [
    #         "'hawaiian hip hop', 'rap'",
    #        "'chicago rap', 'dance pop', 'pop', 'pop rap', 'r&b', 'southern hip hop', 'trap', 'urban contemporary'",
    #        "'pop', 'pop r&b'", "'dance pop', 'pop', 'r&b'",
    #        "'chill r&b', 'pop', 'pop r&b', 'r&b', 'urban contemporary'"
    #     ], 
    #     'artist_name_pl': [
    #         'Russ', 'Jeremih', 'Khalid', 'Beyonc\xc3\xa9','William Singe'
    #     ], 
    #     'artist_pop_can': 82.0, 
    #     'description_pl': '', 
    #     'duration_ms_songs_pl': [
    #         237506.0, 217200.0, 219080.0, 226400.0, 121739.0
    #     ], 
    #     'n_songs_pl': 8.0, 
    #     'name': 'Lit Tunes ', 
    #     'num_albums_pl': 8.0, 
    #     'num_artists_pl': 8.0, 
    #     'track_name_pl': [
    #         'Losin Control', 'Paradise', 'Location','Crazy In Love - Remix', 'Pony'
    #     ], 
    #     'track_pop_pl': [
    #         79.0, 58.0, 83.0, 71.0, 57.0
    #     ],
    #     'duration_ms_seed_pl': 51023.1,
    #     'pid': 1,
    #     'track_uri_pl': [
    #         'spotify:track:4cxMGhkinTocPSVVKWIw0d',
    #         'spotify:track:1wNEBPo3nsbGCZRryI832I',
    #         'spotify:track:152lZdxL1OR0ZMW6KquMif',
    #         'spotify:track:2f4IuijXLxYOeBncS60GUD',
    #         'spotify:track:4Lj8paMFwyKTGfILLELVxt'
    #     ]
    # }
    start = time.process_time()

    playlist_emb = endpoint.predict(instances=[test_instances_dict])
    
    end = time.process_time()
    
    elapsed_time = end - start
    elapsed_time = round(elapsed_time, 2)
    logging.info(f'Deployed query model latency: {elapsed_time} seconds')
    logging.info(f'query embeddings: {playlist_emb.predictions}')
    
    metrics.log_metric("endpoint latency", elapsed_time)
