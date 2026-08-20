"""Helpers for reading per-object ReID tensor metadata."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import numpy as np

BufferReader = Callable[[object, int], np.ndarray]


def l2_normalize(embedding: np.ndarray) -> np.ndarray | None:
    """Return a detached unit vector, or ``None`` for unusable embeddings."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    vector /= norm
    return vector


def _read_layer_buffer(layer: object, feature_size: int, pyds_module: object) -> np.ndarray:
    pointer = pyds_module.get_ptr(layer.buffer)
    c_buffer = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(c_buffer, shape=(feature_size,)).copy()


def _matches_embedding_contract(
    layer: object,
    *,
    feature_size: int,
    output_layer_name: str,
    pyds_module: object,
) -> bool:
    layer_name = getattr(layer, "layerName", None)
    if isinstance(layer_name, bytes):
        layer_name = layer_name.decode(errors="replace")
    infer_dims = getattr(layer, "inferDims", None)
    float_type = getattr(getattr(pyds_module, "NvDsInferDataType", None), "FLOAT", None)
    buffer = getattr(layer, "buffer", None)
    return (
        layer_name == output_layer_name
        and infer_dims is not None
        and int(getattr(infer_dims, "numElements", -1)) == feature_size
        and float_type is not None
        and getattr(layer, "dataType", None) == float_type
        and not bool(getattr(layer, "isInput", True))
        and buffer is not None
        and buffer != 0
    )


def extract_reid_embedding(
    obj_meta: object,
    *,
    pyds_module: object | None = None,
    gie_id: int = 2,
    feature_size: int = 256,
    output_layer_name: str = "fc_pred",
    buffer_reader: BufferReader | None = None,
) -> np.ndarray | None:
    """Read and normalize the selected SGIE output attached to an object."""
    if pyds_module is None:
        import pyds as pyds_module

    tensor_meta_type = pyds_module.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
    node = getattr(obj_meta, "obj_user_meta_list", None)
    while node is not None:
        try:
            user_meta = pyds_module.NvDsUserMeta.cast(node.data)
            if user_meta.base_meta.meta_type == tensor_meta_type:
                tensor_meta = pyds_module.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data
                )
                if int(tensor_meta.unique_id) == gie_id:
                    if int(getattr(tensor_meta, "num_output_layers", 0)) < 1:
                        return None
                    layer = pyds_module.get_nvds_LayerInfo(tensor_meta, 0)
                    if not _matches_embedding_contract(
                        layer,
                        feature_size=feature_size,
                        output_layer_name=output_layer_name,
                        pyds_module=pyds_module,
                    ):
                        return None
                    if buffer_reader is None:
                        raw = _read_layer_buffer(layer, feature_size, pyds_module)
                    else:
                        raw = buffer_reader(layer, feature_size)
                    if np.asarray(raw).size != feature_size:
                        return None
                    return l2_normalize(raw)
            node = node.next
        except StopIteration:
            break
    return None


def extract_frame_reid_embeddings(
    frame_meta: object,
    *,
    pyds_module: object | None = None,
    gie_id: int = 2,
    feature_size: int = 256,
    buffer_reader: BufferReader | None = None,
) -> dict[int, np.ndarray]:
    """Return normalized SGIE vectors keyed by source-local tracker ID."""
    if pyds_module is None:
        import pyds as pyds_module

    embeddings: dict[int, np.ndarray] = {}
    node = getattr(frame_meta, "obj_meta_list", None)
    while node is not None:
        try:
            obj_meta = pyds_module.NvDsObjectMeta.cast(node.data)
            embedding = extract_reid_embedding(
                obj_meta,
                pyds_module=pyds_module,
                gie_id=gie_id,
                feature_size=feature_size,
                buffer_reader=buffer_reader,
            )
            if embedding is not None:
                embeddings[int(obj_meta.object_id)] = embedding
            node = node.next
        except StopIteration:
            break
    return embeddings
