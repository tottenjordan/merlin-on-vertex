
try:
    import cudf
except ImportError:
    cudf = None
import pandas as pd
import tensorflow as tf
from typing import Dict
from merlin.io import Dataset
import itertools


def cupy_array_to_tensor(array):
    return tf.experimental.dlpack.from_dlpack(array.reshape(-1, 1).toDlpack())

def numpy_array_to_tensor(array):
    return tf.convert_to_tensor(array.reshape(-1, 1))

def cudf_series_to_tensor(col) -> tf.Tensor:
    "Convert a cudf.Series to a TensorFlow Tensor with DLPack"
    if isinstance(col.dtype, cudf.ListDtype):
        values = col.list.leaves.values
        offsets = col.list._column.offsets.values
        row_lengths = offsets[1:] - offsets[:-1]
        return cupy_array_to_tensor(values), cupy_array_to_tensor(row_lengths)
    else:
        return cupy_array_to_tensor(col.values)

def pandas_series_to_tensor(col) -> tf.Tensor:
    if len(col) and pd.api.types.is_list_like(col.values[0]):
        values = pd.Series(itertools.chain(*col)).values
        row_lengths = col.map(len).values
        return numpy_array_to_tensor(values), numpy_array_to_tensor(row_lengths)
    else:
        return numpy_array_to_tensor(col.values)
        
    
def dataset_to_tensors(dataset: Dataset) -> Dict[str, tf.Tensor]:
    """Convert a DataFrame to Dict of Tensors"""
    df = dataset.to_ddf().compute()
    if isinstance(df, pd.DataFrame):
        col_to_tensor = pandas_series_to_tensor
    else:
        col_to_tensor = cudf_series_to_tensor
    return {
        column: col_to_tensor(df[column])
        for column in df.columns
    }
