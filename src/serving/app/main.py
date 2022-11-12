from fastapi import FastAPI, Request

import json
import numpy as np
import os
import logging
from fastapi_utils.timing import add_timing_middleware, record_timing

from google.cloud import storage
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
test_bucket = 'gs://jt-merlin-scaling'
predictor_instance = Predictor()
<<<<<<< HEAD

loaded_predictor = predictor_instance.load(artifacts_uri = os.path.join(os.environ['AIP_STORAGE_URI'],'query-tower'),
                                          workflow_uri = os.path.join(os.environ['AIP_STORAGE_URI'], 'workflow'))
=======
ARTIFACT_DIR = "gs://jt-merlin-scaling/pipes-2tower-merlin-tf-v5/run-v1-20221110-024710/model-dir"
loaded_predictor = predictor_instance.load(artifacts_uri = ARTIFACT_DIR)  # os.environ['AIP_STORAGE_URI'])
>>>>>>> 59f09107be76251cf04a2e190c9b19fa1f0ec078

app = FastAPI()
add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    outputs = loaded_predictor.predict(instances)
    # outputs = loaded_predictor.predict(preprocessed_inputs)

    return {"predictions": outputs.numpy().tolist()}
