
import os

# =============================================
#           Cloud Storage Directorires
# =============================================
# BUCKET_source = 'spotify-beam-v3'
BUCKET_source ='spotify-million-playlist-dataset'
BUCKET_destin = 'jt-merlin-scaling'
TRAIN_SRC_DIR = 'train_data_parquet'
VALID_SRC_DIR = 'valid_data_parquet'

# =============================================
#           Setup
# =============================================
VERSION = 'latest-16'
APP = 'spotify'
# MODEL_DISPLAY_NAME = f'nvt-prep-last5-{VERSION}'
# WORKSPACE = f'gs://{BUCKET_destin}/{MODEL_DISPLAY_NAME}'
PROJECT_ID = "hybrid-vertex"
REGION = "us-central1"
VERTEX_SA = f"vertex-sa@{PROJECT_ID}.iam.gserviceaccount.com"

# =============================================
#           Artifacts
# =============================================
# MODEL_DISPLAY_NAME = f"nvt-last5-{VERSION}"
# WORKSPACE = f"gs://jt-merlin-scaling/nvt-last5-{VERSION}"
MODEL_DISPLAY_NAME = f'nvt-last5-{VERSION}'
WORKSPACE = f'gs://{BUCKET_destin}/{MODEL_DISPLAY_NAME}'
NVT_IMAGE_URI = "gcr.io/hybrid-vertex/nvt-preprocessing"

# Docker definitions
IMAGE_NAME = 'nvt-preprocessing'
IMAGE_URI = f'gcr.io/{PROJECT_ID}/{IMAGE_NAME}'

# =============================================
#           Pipeline Configs
# =============================================
# PREPROCESS_PARQUET_PIPELINE_NAME = f"nvt-parquet-{VERSION}"
# PREPROCESS_PARQUET_PIPELINE_ROOT = f"gs://jt-merlin-scaling/{MODEL_DISPLAY_NAME}/{PREPROCESS_PARQUET_PIPELINE_NAME}"
PREPROCESS_PARQUET_PIPELINE_NAME = f'nvt-parquet-{VERSION}'
PREPROCESS_PARQUET_PIPELINE_ROOT = os.path.join(WORKSPACE, PREPROCESS_PARQUET_PIPELINE_NAME)

# 4 tesla 4s
# INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "n1-highmem-64")
# CPU_LIMIT = os.getenv("CPU_LIMIT", "64")
# MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "624G")
# GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
# GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_T4")

# 2 A100s
# INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "a2-highgpu-2g")
# CPU_LIMIT = os.getenv("CPU_LIMIT", "24")
# MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "170G")
# GPU_LIMIT = os.getenv("GPU_LIMIT", "2")
# GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_A100")

# 4 A100s
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "a2-highgpu-4g")
CPU_LIMIT = os.getenv("CPU_LIMIT", "48")
MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "340G")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_A100")
