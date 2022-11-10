# merlin-on-vertex
### guidance and code examples for scaling deep retrieval workloads using NVIDIA's [Merlin](https://github.com/NVIDIA-Merlin/Merlin) and [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) ecosystems on Google Cloud's [Vertex AI platform](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform)

> [Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified Machine Learning platform to help data scientists and machine learning engineers increase experimentation, deploy faster, and manage models with confidence.

> [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is an open-source framework for building large-scale deep learning recommender system.

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

### The dataset

> Spotify's Million Playlist Dataset (MPD) - see [here](https://github.com/jswortz/spotify_mpd_two_tower/blob/cbbd29fd71e8b500683635a19f0aa8ae657db884/00-bq-data-prep.ipynb) for downloading and preparing the dataset in BigQuery

**citation:**

*Ching-Wei Chen, Paul Lamere, Markus Schedl, and Hamed Zamani. Recsys Challenge 2018: Automatic Music Playlist Continuation. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys â€™18), 2018.*

### Data preprocessing pipelines





