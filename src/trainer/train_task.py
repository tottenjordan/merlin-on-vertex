
import argparse
import json
import logging
import os
import sys
import time
import pandas as pd

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
# os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory

# merlin
# from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.io.dataset import Dataset as MerlinDataset
from merlin.models.tf.outputs.base import DotProduct, MetricsFn, ModelOutput
from merlin.schema.tags import Tags
import merlin.models.tf as mm

from merlin.models.utils.dataset import unique_rows_by_features

# nvtabular
import nvtabular as nvt
import nvtabular.ops as ops

# tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib

# gcp
import google.cloud.aiplatform as vertex_ai
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob
# import hypertune
# from google.cloud.aiplatform.training_utils import cloud_profiler

# repo
from .two_tower_model import create_two_tower
from .train_utils import (
    get_upload_logs_to_manged_tb_command, 
    get_arch_from_string, 
    _upload_blob_gcs, 
    upload_from_directory
)

# local
HYPERTUNE_METRIC_NAME = 'AUC'
LOCAL_MODEL_DIR = '/tmp/saved_model'
LOCAL_CHECKPOINT_DIR = '/tmp/checkpoints'
# ====================================================
# TRAINING SCRIPT
# ====================================================
    
def main(args):
    """Runs a training loop."""
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # tf.debugging.set_log_device_placement(True) # logs all tf ops and their device placement;
    # os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    # os.environ['TF_GPU_THREAD_COUNT']='1'
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    # TB_RESOURCE_NAME = f'{args.tb_name}'
    
    logging.info(f"EXPERIMENT_NAME: {args.experiment_name}\n RUN_NAME: {args.experiment_run}")
    logging.info(f'TB_RESOURCE_NAME tb_name: {args.tb_name}')
    
    # ====================================================
    # Set directories
    # ====================================================
    WORKING_DIR_GCS_URI = f'gs://{args.train_output_bucket}/{args.experiment_name}/{args.experiment_run}'
    logging.info(f"WORKING_DIR_GCS_URI: {WORKING_DIR_GCS_URI}")
    
    LOGS_DIR = f'{WORKING_DIR_GCS_URI}/tb_logs'
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        LOGS_DIR=os.environ['AIP_TENSORBOARD_LOG_DIR']
        logging.info(f'AIP_TENSORBOARD_LOG_DIR: {LOGS_DIR}')
        
    logging.info(f'TensorBoard LOGS_DIR: {LOGS_DIR}')
    
    # ====================================================
    # Init Clients
    # ====================================================
    project_number = os.environ["CLOUD_ML_PROJECT_ID"]
    storage_client = storage.Client(project=f'{args.project}')
    vertex_ai.init(
        project=f'{args.project}',
        location=f'{args.location}',
        experiment=f'{args.experiment_name}',
    )
    
    logging.info("vertex_ai initialized...")
    
    # ====================================================
    # Set Device / GPU Strategy
    # ====================================================    
    logging.info("Detecting devices....")
    logging.info(f'Detected Devices {str(device_lib.list_local_devices())}')
    
    logging.info("Setting device strategy...")
    
    # Single Machine, single compute device
    if args.distribute == 'single':
        if tf.test.is_gpu_available(): # TODO: replace with - tf.config.list_physical_devices('GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")
    
    # Single Machine, multiple compute device
    elif args.distribute == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")

    # Multi Machine, multiple compute device
    elif args.distribute == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
        
    
    # set related vars...
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = NUM_WORKERS * args.per_gpu_batch_size
    logging.info(f'NUM_WORKERS = {NUM_WORKERS}')
    logging.info(f'GLOBAL_BATCH_SIZE: {GLOBAL_BATCH_SIZE}')
    
    # set worker vars...
    logging.info(f'Setting task_type and task_id...')
    if args.distribute == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
    
    logging.info(f'task_type = {task_type}')
    logging.info(f'task_id = {task_id}')
        
    # ====================================================
    # Prepare Train and Valid Data
    # ====================================================
    logging.info(f'Loading workflow & schema from : {args.workflow_dir}')
    
    workflow = nvt.Workflow.load(args.workflow_dir)
    schema = workflow.output_schema
    
    train_data = MerlinDataset(os.path.join(args.train_dir, "*.parquet"), schema=schema, part_size="1GB")
    valid_data = MerlinDataset(os.path.join(args.valid_dir, "*.parquet"), schema=schema, part_size="1GB")
    
    # ====================================================
    # Callbacks
    # ====================================================
    class UploadTBLogsBatchEnd(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            os.system(
                get_upload_logs_to_manged_tb_command(
                    tb_resource_name=args.tb_name, 
                    logs_dir=LOGS_DIR, 
                    experiment_name=args.experiment_name,
                    ttl_hrs = 5, 
                    oneshot="true",
                )
            )
            
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=0, 
        write_graph=True, 
        # profile_batch=(20,50)
    )
    
    # ====================================================
    # Train
    # ==================================================== 
    LAYER_SIZES = get_arch_from_string(args.layer_sizes)
    logging.info(f'LAYER_SIZES: {LAYER_SIZES}')

    # with strategy.scope():
    model = create_two_tower(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        workflow_dir=args.workflow_dir,
        layer_sizes=LAYER_SIZES # args.layer_sizes,
    )
        
        
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(args.learning_rate),
        run_eagerly=False,
        metrics=[mm.RecallAt(1), mm.RecallAt(10), mm.NDCGAt(10)],
    )
    
    # cloud_profiler.init() # managed TB profiler
        
    logging.info('Starting training loop...')
    
    start_model_fit = time.time()
    
    model.fit(
        train_data, 
        validation_data=valid_data, 
        batch_size=GLOBAL_BATCH_SIZE, 
        epochs=args.num_epochs,
        # steps_per_epoch=20, 
        callbacks=[
            tensorboard_callback, 
            UploadTBLogsBatchEnd()
        ],
    )
    
    # capture elapsed time
    end_model_fit = time.time()
    elapsed_model_fit = end_model_fit - start_model_fit
    elapsed_model_fit = round(elapsed_model_fit, 2)
    logging.info(f'Elapsed model_fit: {elapsed_model_fit} seconds')
    
    # ====================================================
    # metaparams & metrics for Vertex Ai Experiments
    # ====================================================
    logging.info('Logging params & metrics for Vertex Experiments')
    
    # get the metrics for the experiment run
    history_keys = model.history.history.keys()
    metrics_dict = {}
    _ = [metrics_dict.update({key: model.history.history[key][-1]}) for key in history_keys]
    metrics_dict["elapsed_model_fit"] = elapsed_model_fit 
    logging.info(f'metrics_dict: {metrics_dict}')
    
    metaparams = {}
    metaparams["experiment_name"] = f'{args.experiment_name}'
    metaparams["experiment_run"] = f"{args.experiment_run}"
    logging.info(f'metaparams: {metaparams}')
    
    hyperparams = {}
    hyperparams["epochs"] = int(args.num_epochs)
    hyperparams["num_gpus"] = NUM_WORKERS # num_gpus
    hyperparams["per_gpu_batch_size"] = args.per_gpu_batch_size
    hyperparams["global_batch_size"] = GLOBAL_BATCH_SIZE
    hyperparams["learning_rate"] = args.learning_rate
    hyperparams['layers'] = f'{args.layer_sizes}'
    logging.info(f'hyperparams: {hyperparams}')
    
    # ====================================================
    # Experiments
    # ====================================================
    logging.info(f"Creating run: {args.experiment_run}; for experiment: {args.experiment_name}")
    
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
    
        # Create experiment
        vertex_ai.init(experiment=args.experiment_name)

        with vertex_ai.start_run(args.experiment_run) as my_run:
            logging.info(f"logging metrics_dict")
            my_run.log_metrics(metrics_dict)

            logging.info(f"logging metaparams")
            my_run.log_params(metaparams)

            logging.info(f"logging hyperparams")
            my_run.log_params(hyperparams)
        
    # =============================================
    # save retrieval (query) tower
    # =============================================
    QUERY_TOWER_LOCAL_DIR = 'query_tower'
    CANDIDATE_TOWER_LOCAL_DIR = 'candidate_tower'
    # set vars...
    MODEL_DIR = f"{WORKING_DIR_GCS_URI}/model_dir"
    logging.info(f'Saving towers to {MODEL_DIR}')
    
    QUERY_TOWER_PATH = f"{MODEL_DIR}/query_tower"
    CANDIDATE_TOWER_PATH = f"{MODEL_DIR}/candidate_tower"
    EMBEDDINGS_PATH = f"{MODEL_DIR}/candidate_embeddings"
    
    if task_type == 'chief':
        
        # save query tower
        query_tower = model.query_encoder
        query_tower.save(f'{QUERY_TOWER_LOCAL_DIR}/')
        logging.info(f'Saved query tower locally to {QUERY_TOWER_LOCAL_DIR}')
        upload_from_directory(f'./{QUERY_TOWER_LOCAL_DIR}', args.train_output_bucket, f'{args.experiment_name}/{args.experiment_run}/model_dir', f'{args.project}')
        logging.info(f'Saved query tower to {QUERY_TOWER_PATH}')
        
        candidate_tower = model.candidate_encoder
        candidate_tower.save(f'{CANDIDATE_TOWER_LOCAL_DIR}')
        logging.info(f'Saved candidate tower locally to {CANDIDATE_TOWER_LOCAL_DIR}')
        upload_from_directory(f'./{CANDIDATE_TOWER_LOCAL_DIR}', args.train_output_bucket, f'{args.experiment_name}/{args.experiment_run}/model_dir', f'{args.project}')
        logging.info(f'Saved candidate tower to {CANDIDATE_TOWER_PATH}')

    
    # =============================================
    # save embeddings for ME index
    # =============================================
    EMBEDDINGS_FILE_NAME = "candidate_embeddings.json"
    logging.info(f"Saving {EMBEDDINGS_FILE_NAME} to {EMBEDDINGS_PATH}")
    
    def format_for_matching_engine(data) -> None:
        cols = [str(i) for i in range(LAYER_SIZES[-1])]      # ensure we are only pulling 0-EMBEDDING_DIM cols
        emb = [data[col] for col in cols]                    # get the embeddings
        formatted_emb = '{"id":"' + str(data['track_uri_can']) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}'
        with open(f"{EMBEDDINGS_FILE_NAME}", 'a') as f:
            f.write(formatted_emb)
            f.write("\n")
    
    # !rm candidate_embeddings.json > /dev/null 
    # !touch candidate_embeddings.json
    item_data = pd.read_parquet(f'{args.workflow_dir}/categories/unique.track_uri_can.parquet')
    lookup_dict = dict(item_data['track_uri_can'])

    # item embeds from TRAIN
    start_embeds = time.time()
    
    item_features = (unique_rows_by_features(train_data, Tags.ITEM, Tags.ID))
    item_embs = model.candidate_embeddings(item_features, index=item_features.schema['track_uri_can'], batch_size=10000)
    item_emb_pd = item_embs.compute().to_pandas().fillna(1e-10).reset_index() #filling blanks with an epsilon value
    item_emb_pd['track_uri_can'] = item_emb_pd['track_uri_can'].apply(lambda l: lookup_dict[l])
    _ = item_emb_pd.apply(format_for_matching_engine, axis=1)
    
    # capture elapsed time
    end_embeds = time.time()
    elapsed_time = end_embeds - start_embeds
    elapsed_time = round(elapsed_time, 2)
    logging.info(f'Elapsed time writting TRAIN embeddings: {elapsed_time} seconds')
    
    # item embeds from VALID
    start_embeds = time.time()
    
    item_features_val = (unique_rows_by_features(valid_data, Tags.ITEM, Tags.ID))
    item_embs_val = model.candidate_embeddings(item_features_val, index=item_features_val.schema['track_uri_can'], batch_size=10000)
    item_emb_pd_val = item_embs_val.compute().to_pandas().fillna(1e-10).reset_index() #filling blanks with an epsilon value
    item_emb_pd_val['track_uri_can'] = item_emb_pd_val['track_uri_can'].apply(lambda l: lookup_dict[l])
    _ = item_emb_pd_val.apply(format_for_matching_engine, axis=1)
    
    # capture elapsed time
    end_embeds = time.time()
    elapsed_time = end_embeds - start_embeds
    elapsed_time = round(elapsed_time, 2)
    logging.info(f'Elapsed time writting VALID embeddings: {elapsed_time} seconds')
    
    if task_type == 'chief':
        _upload_blob_gcs(
            EMBEDDINGS_PATH, 
            f"{EMBEDDINGS_FILE_NAME}", 
            f"{EMBEDDINGS_FILE_NAME}",
            args.project
        )
    
    logging.info('All done - model saved') #all done
    
# ====================================================
# arg parser
# ====================================================
    
def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=False,
        default='unnamed-experiment',
        help='name of vertex ai experiment'
    )
    parser.add_argument(
        '--experiment_run',
        type=str,
        required=False,
        default='unnamed_run',
        help='name of vertex ai experiment run'
    )
    parser.add_argument(
        '--tb_name',
        type=str,
        required=True,
        help='projects/XXXXXX/locations/us-central1/tensorboards/XXXXXXXX'
    )
    parser.add_argument(
        '--distribute',
        type=str,
        required=False,
        default='single',
        help='training strategy'
    )
    parser.add_argument(
        '--train_output_bucket',
        type=str,
        required=True,
        # default='single',
        help='gcs bucket name'
    )
    parser.add_argument(
        '--workflow_dir',
        type=str,
        required=True,
        help='Path to saved workflow.pkl e.g., nvt-analyzed'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        required=True,
        help='Path to training data _file_list.txt'
    )
    parser.add_argument(
        '--valid_dir',
        type=str,
        required=True,
        help='Path to validation data _file_list.txt'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True,
        help='num_epochs'
    )
    parser.add_argument(
        '--per_gpu_batch_size',
        type=int,
        required=True,
        help='Per GPU Batch size'
    )
    parser.add_argument(
        '--layer_sizes',
        type=str,
        required=False,
        default='[512, 256, 128]',
        help='layer_sizes'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        required=False,
        default=.001,
        help='learning_rate'
    )
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='gcp project'
    )
    parser.add_argument(
        '--location',
        type=str,
        required=True,
        help='gcp location'
    )
    # parser.add_argument(
    #     '--gpus',
    #     type=str,
    #     required=False,
    #     default='[[0]]',
    #     help='GPU devices to use for Preprocessing'
    # )
    
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )

    parsed_args = parse_args()

    # parsed_args.gpus = json.loads(parsed_args.gpus)

    # parsed_args.slot_size_array = [
    #     int(i) for i in parsed_args.slot_size_array.split(sep=' ')
    # ]

    logging.info('Args: %s', parsed_args)
    start_time = time.time()
    logging.info('Starting training')

    main(parsed_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info('Training completed. Elapsed time: %s', elapsed_time )
