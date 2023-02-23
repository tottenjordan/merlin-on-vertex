
from typing import List, Any

import nvtabular as nvt
# # import nvtabular.ops as ops

# from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags
import merlin.models.tf as mm
from merlin.models.tf.outputs.base import DotProduct, MetricsFn, ModelOutput

import logging

import tensorflow as tf


def create_two_tower(
    train_dir: str,
    valid_dir: str,
    workflow_dir: str,
    layer_sizes: List[Any] = [512, 256, 128],
):
    
    #=========================================
    # get workflow details
    #=========================================
    workflow = nvt.Workflow.load(workflow_dir) # gs://spotify-merlin-v1/nvt-preprocessing-spotify-v24/nvt-analyzed
    
    schema = workflow.output_schema
    # embeddings = ops.get_embedding_sizes(workflow)
    
    user_schema = schema.select_by_tag(Tags.USER)
    user_inputs = mm.InputBlockV2(user_schema)
    
    #=========================================
    # build towers
    #=========================================
    query = mm.Encoder(user_inputs, mm.MLPBlock(layer_sizes))
    
    item_schema = schema.select_by_tag(Tags.ITEM)
    item_inputs = mm.InputBlockV2(
        item_schema,
    )
    candidate = mm.Encoder(item_inputs, mm.MLPBlock(layer_sizes))
    
    model = mm.TwoTowerModelV2(
        query_tower=query,
        candidate_tower=candidate,
        # output=mm.ContrastiveOutput(
        #     to_call=DotProduct(),
        #     negative_samplers="in-batch",
        #     schema=item_schema.select_by_tag(Tags.ITEM_ID),
        #     candidate_name="item",
        # )
    )
    
    return model
