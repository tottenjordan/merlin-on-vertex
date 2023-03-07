
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.18.1',
        # 'google-cloud-storage',
    ],
)
def upload_custom_model(
    project: str,
    location: str,
    version: str,
    display_name: str,
    artifact_uri: str,
    unmanaged_container_model: Input[Artifact],
    serving_container_image_uri: str,
) -> NamedTuple('Outputs', [
    ('model', Artifact),
    ('model_resource_name', str),
]):
    
    import logging
    from google.cloud import aiplatform as vertex_ai

    vertex_ai.init(
        project=project,
        location=location,
    )
    logging.info(f" display_name: {display_name}")
    logging.info(f" artifact_uri: {artifact_uri}")
    logging.info(f" unmanaged_container_model: {unmanaged_container_model}")
    logging.info(f" serving_container_image_uri: {serving_container_image_uri}")
    
    logging.info(f"Uploading model to Vertex...")
    model = vertex_ai.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route='/predict',
        serving_container_health_route='/health',
        serving_container_command=["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $AIP_HTTP_PORT"],
        serving_container_args='--gpus all',
        # parent_model=PARENT_MODEL,
        sync=True,
    )
    
    MODEL_RESOURCE_NAME = model.resource_name
    logging.info(f" MODEL_RESOURCE_NAME: {MODEL_RESOURCE_NAME}")
    
    return (
        model,
        f'{MODEL_RESOURCE_NAME}',
    )
