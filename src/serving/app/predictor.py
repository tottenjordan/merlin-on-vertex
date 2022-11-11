import nvtabular as nvt
import pandas as pd
import os
import json
import merlin.models.tf as mm
from nvtabular.loader.tf_utils import configure_tensorflow
configure_tensorflow()
import tensorflow as tf
import time
import logging


# These are helper functions that ensure the dictionary input is in a certain order and types are preserved
# this is to get scalar values to appear first in the dict to not confuse pandas with lists https://github.com/pandas-dev/pandas/issues/46092
reordered_keys = [
    'collaborative', 
    'album_name_pl', 
    'artist_genres_pl', 
    'artist_name_pl', 
    'artist_pop_can', 
    'description_pl', 
    'duration_ms_songs_pl', 
    'n_songs_pl', 
    'name', 
    'num_albums_pl', 
    'num_artists_pl', 
    'track_name_pl', 
    'track_pop_pl', 
    'duration_ms_seed_pl', 
    'pid', 
    'track_uri_pl'
]

float_num_fix = ['n_songs_pl','num_albums_pl','num_artists_pl','duration_ms_seed_pl']
float_list_fix = ['track_pop_pl', 'duration_ms_songs_pl']
    
def fix_list_num_dtypes(num_list):
    "this fixes lists of ints to list of floats converted in json input"
    return [float(x) for x in num_list]

def fix_num_dtypes(num):
    "this fixes ints and casts to floats"
    return float(num)

def fix_types(k, v):
    if k in float_num_fix:
        return fix_num_dtypes(v)
    if k in float_list_fix:
        return fix_list_num_dtypes(v)
    else:
        return v

def create_pandas_instance(inputs):
    """
    Helper function to reorder the input to have a sclar first for pandas
    And fix the types converted when data is imported by fastAPI
    """
    if type(inputs) == list:
        header = inputs[0]
        reordered_header_dict = {k: fix_types(k,header[k]) for k in reordered_keys}
        pandas_instance = pd.DataFrame.from_dict(reordered_header_dict, orient='index').T
        if len(inputs) > 1:
            for ti in inputs[1:]:
                reordered_dict = {k: fix_types(k,ti[k]) for k in reordered_keys}
                pandas_instance = pandas_instance.append(pd.DataFrame.from_dict(reordered_dict, orient='index').T)
    else:
        reordered_dict = {k: fix_types(k,inputs[k]) for k in reordered_keys}
        pandas_instance = pd.DataFrame.from_dict(reordered_dict, orient='index').T
    return pandas_instance

class Predictor():
    """Interface of the Predictor class for Custom Prediction Routines.
    The Predictor is responsible for the ML logic for processing a prediction request.
    Specifically, the Predictor must define:
    (1) How to load all model artifacts used during prediction into memory.
    (2) The logic that should be executed at predict time.
    When using the default PredictionHandler, the Predictor will be invoked as follows:
      predictor.postprocess(predictor.predict(predictor.preprocess(prediction_input)))
    """
    def __init__(self):
        return
    
    def load(self, artifacts_uri):
        """Loads the model artifact.
        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
        """
        logging.info("loading model and workflow")
        start = time.process_time()
        
        self.model = tf.keras.models.load_model(os.path.join(artifacts_uri, "query-tower"))
        self.workflow = nvt.Workflow.load(os.path.join(artifacts_uri, "workflow"))
        self.workflow = self.workflow.remove_inputs(
            [
                'track_pop_can', 
                'track_uri_can', 
                'duration_ms_can', 
                'track_name_can', 
                'artist_name_can',
                'album_name_can',
                'album_uri_can',
                'artist_followers_can', 
                'artist_genres_can',
                'artist_name_can', 
                'artist_pop_can',
                'artist_pop_pl',
                'artist_uri_can', 
                'artists_followers_pl'
            ]
        )
        self.loader = None # will load this after first load
        self.n_rows = 0
        logging.info(f"loading took {time.process_time() - start} seconds")
        
        return self

    
    def predict(self, prediction_input):
        """Preprocesses the prediction input before doing the prediction.
        Args:
            prediction_input (Any):
                Required. The prediction input that needs to be preprocessed.
        Returns:
            The preprocessed prediction input.
        """
        # handle different input types, can take a dict or list of dicts
        self.n_rows = len(prediction_input)
        
        # pandas convert
        start = time.process_time()
        pandas_instance = create_pandas_instance(prediction_input[0])
        logging.info(f"Pandas conversion took {time.process_time() - start} seconds")
        
        # nvtabular data loading
        start = time.process_time()
        transformed_inputs = nvt.Dataset(pandas_instance)
        logging.info(f"NVT data loading took {time.process_time() - start} seconds")
        
        # workflow transformation
        start = time.process_time()
        transformed_instance = self.workflow.transform(transformed_inputs)
        logging.info(f"Workflow transformation took {time.process_time() - start} seconds")

        # tensorflow data loader
        start = time.process_time()
        batch = mm.sample_batch(transformed_instance, batch_size=1, include_targets=False, shuffle=False)
        logging.info(f"TF Dataloader took {time.process_time() - start} seconds")
        
        # model predict
        start = time.process_time()
        output = self.model(batch)
        logging.info(f"Prediction took {time.process_time() - start} seconds")
        
        return output
