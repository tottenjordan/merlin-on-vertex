from fastapi import FastAPI, Request

import json
import numpy as np
import os
import logging


from google.cloud import storage
from predictor import Predictor

app = FastAPI()

predictor_instance = Predictor()
loaded_predictor = predictor_instance.load(artifacts_uri = os.environ['AIP_STORAGE_URI'])

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    outputs = loaded_predictor.predict(instances)

    return {"predictions": outputs[1].numpy().tolist()}
