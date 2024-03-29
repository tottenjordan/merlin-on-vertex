# merlin-on-vertex


### scaling deep retrieval workloads using NVIDIA's [Merlin](https://github.com/NVIDIA-Merlin/Merlin) & [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) on Google Cloud's [Vertex AI platform](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform)

> [Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified Machine Learning platform to help data scientists and machine learning engineers increase experimentation, deploy faster, and manage models with confidence.

> [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is an open-source framework for building large-scale deep learning recommender system.

![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/deep-retrieval-workflow.png)

See [this repo](https://github.com/jswortz/spotify_mpd_two_tower/tree/cbbd29fd71e8b500683635a19f0aa8ae657db884) for a sample development workflow 

---
### Repo structure

* [01-data-preprocess-pipeline.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/01-data-preprocess-pipeline.ipynb)  
> * launch Vertex pipeline to orchestrate GPU-based data preprocessing with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)

* [02-merlin-vertex-training.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/02-merlin-vertex-training.ipynb) 
> * build two-tower encoder with [Merlin Models](https://github.com/NVIDIA-Merlin/models),
> * prepare training application (container/image) with [Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image)
> * scale training with Vertex AI and A100 GPU  

* [03a-query-model-inference.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/03a-query-model-inference.ipynb) 
> * prepare serving application (container/image) with [Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image) 
> * deploy trained query tower to [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/overview) endpoint
> * [03b-build-docker.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/03b-build-docker.ipynb) - (optionally) prepare serving application with *docker*

* [04-matching-engine.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/04-matching-engine.ipynb)
> * Use candidate embeddings generated from Vertex Train job to create Matching Engine serving index
> * create and deploy ANN and brute-force indexes
> * compute recall and retrieval latency

* [05-train-deploy-pipeline.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/05-train-deploy-pipeline.ipynb) 
> * orchestrate e2e model training and deployemnt (notebooks 02 and 03)
> * create candidate index and deploy to index endpoint with [Vertex AI Matching Engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)

* [06-recs-for-your-spotify.ipynb](https://github.com/tottenjordan/merlin-on-vertex/blob/main/06-recs-for-your-spotify.ipynb) 
> * using trained towers and deployed Matching Engine index, generate playlist recommendations for your own (or any public) Spotify playlist(s) 

The Python modules are in the `src` folder:
* `src/preprocessor` - data preprocessing utility functions and classes
* `src/process_pipes` - vertex pipeline components for orchestrating data preprocessing
* `src/serving/app` - deployment and serving utility functions and classes
* `src/train_pipes` - vertex pipeline components for orchestrating the training and deployment pipeline
* `src/trainer` - model definitions and training application

---

### Objectives
* Operationalize large scale data preprocessing pipelines with NVTabular and [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
* Scaling model training with [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
* Building and deploying a scalable approximate nearest neighbor (ANN) index using [Vertex AI Matching Engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)
> Deploying trained models and serving predictions with [Vertex Prediction](https://cloud.google.com/vertex-ai/docs/predictions/getting-predictions) (Triton server coming soon)

### TODOs
* Serving deployment with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) on [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/overview)
* Add ranking model workflow and demonstrate end-to-end retrieval -> ranking deployment
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

![alt text](https://github.com/tottenjordan/merlin-on-vertex/blob/main/imgs/merlin-e2e-pipe.png)

* Build custom containers for training and serving
* Train Merlin retrieval model
* Import Query and Candidate Towers to pipeline DAG
* Register and deploy Query Tower with Vertex AI
* Create Matching Engine indexes amd index endpoints
* Deploy indexes to index endpoints 





