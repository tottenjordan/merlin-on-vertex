# merlin-on-vertex


### scaling deep retrieval workloads using NVIDIA's [Merlin](https://github.com/NVIDIA-Merlin/Merlin) & [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) on Google Cloud's [Vertex AI platform](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform)

> [Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified Machine Learning platform to help data scientists and machine learning engineers increase experimentation, deploy faster, and manage models with confidence.

> [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is an open-source framework for building large-scale deep learning recommender system.

### Repo structure

> **TODO**

---

![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/deep-retrieval-workflow.png)

See [this repo](https://github.com/jswortz/spotify_mpd_two_tower/tree/cbbd29fd71e8b500683635a19f0aa8ae657db884) for a sample development workflow 

### Objectives
* Operationalize large scale data preprocessing pipelines with NVTabular and [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
* Scaling model training with [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
* Building and deploying a scalable approximate nearest neighbor (ANN) index using [Vertex AI Matching Engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)
> Deploying trained models and serving predictions with [Vertex Prediction](https://cloud.google.com/vertex-ai/docs/predictions/getting-predictions) (Triton server coming soon)

### TODOs
* Query tower's custom serving image
* Serving deployment with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) on [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/overview)
* Add ranking model workflow and demonstrate end-to-end retrieval -> ranking deployment
* Add data prep instructions and discussion
---

### The dataset

Spotify's Million Playlist Dataset (MPD) - see [here](https://github.com/jswortz/spotify_mpd_two_tower/blob/cbbd29fd71e8b500683635a19f0aa8ae657db884/00-bq-data-prep.ipynb) for downloading and preparing the dataset in BigQuery

*Ching-Wei Chen, Paul Lamere, Markus Schedl, and Hamed Zamani. Recsys Challenge 2018: Automatic Music Playlist Continuation. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys â€™18), 2018.*

---

### Data preprocessing pipeline

![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/data-process-pipes.png)

* Create and save training and validation splits
* Store data split statistics and schema (NVTabular's `workflow`)
* Transform data splits and prepare them for training and serving tasks
* Orchestrate these NVTabular pipelines with Vertex Managed Pipelines
* Scale pipeline processing tasks with single or multiple GPU configurations

> With 4 Tesla T4 GPUs per processing component, pipeline processes our Spotify MPD in ~27 minutes
---

### Training -> Deployment pipeline

![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/merlin-pipe.png)

* Build custom containers for training and serving
* Train Merlin retrieval model
* Import Query and Candidate Towers to pipeline DAG
* Register and deploy Query Tower with Vertex AI
* Create Matching Engine indexes amd index endpoints
* Deploy indexes to index endpoints 

### two-tower model for deep retrieval 
![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/2tower-diagram.png)





