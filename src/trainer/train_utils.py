
# ====================================================
# Helper functions - moved from train_task.py
# ====================================================

import argparse
import json
import logging
import os
import sys
import time
import pandas as pd

from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob

import glob

# GCS_CLIENT = storage.Client()

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

def get_upload_logs_to_manged_tb_command(tb_resource_name, logs_dir, experiment_name, ttl_hrs, oneshot="false"):
    """
    Run this and copy/paste the command into terminal to have 
    upload the tensorboard logs from this machine to the managed tb instance
    Note that the log dir is at the granularity of the run to help select the proper
    timestamped run in Tensorboard
    You can also run this in one-shot mode after training is done 
    to upload all tb objects at once
    """
    return(
        f"""tb-gcp-uploader --tensorboard_resource_name={tb_resource_name} \
        --logdir={logs_dir} \
        --experiment_name={experiment_name} \
        --one_shot={oneshot} \
        --event_file_inactive_secs={60*60*ttl_hrs}"""
    )

def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name, project):
    """Uploads a file to GCS bucket"""
    storage_client = storage.Client(project=project)
    blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
    blob.bucket._client = storage_client
    blob.upload_from_filename(source_file_name)
    
def get_arch_from_string(arch_string):
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    return [int(x) for x in q.split(',')]

def upload_from_directory(
    directory_path: str, 
    dest_bucket_name: str, 
    dest_blob_name: str,
    project: str,
):
    storage_client = storage.Client(project=project)
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = storage_client.get_bucket(dest_bucket_name)
    
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
