import os

# =============================================
#           Cloud Storage Directorires
# =============================================
BUCKET_source = 'spotify-beam-v3'
BUCKET_destin = 'jt-merlin-scaling'
TRAIN_SRC_DIR = 'train_data_parquet'
VALID_SRC_DIR = 'valid_data_parquet'

# =============================================
#           Setup
# =============================================
VERSION = 'v1full'
APP = 'spotify'
# MODEL_DISPLAY_NAME = f'nvt-prep-last5-{VERSION}'
# WORKSPACE = f'gs://{BUCKET_destin}/{MODEL_DISPLAY_NAME}'
PROJECT_ID = "hybrid-vertex"
REGION = "us-central1"
VERTEX_SA = f"vertex-sa@{PROJECT_ID}.iam.gserviceaccount.com"

# =============================================
#           Artifacts
# =============================================
MODEL_DISPLAY_NAME = "nvt-last5-v1full"
WORKSPACE = "gs://jt-merlin-scaling/nvt-last5-v1full"
NVT_IMAGE_URI = "gcr.io/hybrid-vertex/nvt-preprocessing"

# =============================================
#           Pipeline Configs
# =============================================
PREPROCESS_PARQUET_PIPELINE_NAME = "nvt-parquet-v1full"
PREPROCESS_PARQUET_PIPELINE_ROOT = "gs://jt-merlin-scaling/nvt-last5-v1full/nvt-parquet-v1full"

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "n1-highmem-64")
CPU_LIMIT = os.getenv("CPU_LIMIT", "64")
MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "624G")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_T4")
