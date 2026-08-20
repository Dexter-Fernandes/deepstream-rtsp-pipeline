from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.reid import extract_frame_reid_embeddings, extract_reid_embedding, l2_normalize


@dataclass
class _Node:
    data: object
    next: object = None


class _Cast:
    @staticmethod
    def cast(value):
        return value


def _layer(**overrides):
    values = {
        "buffer": "host-pointer",
        "dataType": 0,
        "inferDims": SimpleNamespace(numElements=256),
        "isInput": False,
        "layerName": "fc_pred",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pyds(layer=None):
    return SimpleNamespace(
        NvDsUserMeta=_Cast,
        NvDsInferTensorMeta=_Cast,
        NvDsInferDataType=SimpleNamespace(FLOAT=0, HALF=1),
        NvDsMetaType=SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=99),
        get_nvds_LayerInfo=lambda _meta, _index: layer or _layer(),
    )


def test_l2_normalize_returns_unit_vector_without_mutating_input():
    raw = np.array([3.0, 4.0], dtype=np.float32)

    normalized = l2_normalize(raw)

    np.testing.assert_allclose(normalized, [0.6, 0.8])
    np.testing.assert_allclose(raw, [3.0, 4.0])


def test_extract_reid_embedding_reads_the_sgie_tensor_for_an_object():
    tensor_meta = SimpleNamespace(unique_id=2, num_output_layers=1)
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(meta_type=99),
        user_meta_data=tensor_meta,
    )
    obj_meta = SimpleNamespace(obj_user_meta_list=_Node(user_meta))
    pyds = _pyds(_layer(inferDims=SimpleNamespace(numElements=2)))

    embedding = extract_reid_embedding(
        obj_meta,
        pyds_module=pyds,
        feature_size=2,
        buffer_reader=lambda _layer, _size: np.array([3.0, 4.0], dtype=np.float32),
    )

    np.testing.assert_allclose(embedding, [0.6, 0.8])


def test_extract_reid_embedding_ignores_other_gie_outputs():
    tensor_meta = SimpleNamespace(unique_id=7, num_output_layers=1)
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(meta_type=99),
        user_meta_data=tensor_meta,
    )
    obj_meta = SimpleNamespace(obj_user_meta_list=_Node(user_meta))
    pyds = _pyds()

    assert extract_reid_embedding(obj_meta, pyds_module=pyds, gie_id=2) is None


def test_zero_embedding_is_rejected_instead_of_producing_nan():
    assert l2_normalize(np.zeros(4, dtype=np.float32)) is None


@pytest.mark.parametrize(
    ("tensor_overrides", "layer_overrides"),
    [
        ({"num_output_layers": 0}, {}),
        ({}, {"layerName": "other"}),
        ({}, {"inferDims": SimpleNamespace(numElements=128)}),
        ({}, {"dataType": 1}),
        ({}, {"buffer": None}),
        ({}, {"isInput": True}),
    ],
)
def test_incompatible_tensor_contract_is_rejected_before_buffer_read(
    tensor_overrides,
    layer_overrides,
):
    tensor_values = {"unique_id": 2, "num_output_layers": 1}
    tensor_values.update(tensor_overrides)
    tensor_meta = SimpleNamespace(**tensor_values)
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(meta_type=99),
        user_meta_data=tensor_meta,
    )
    obj_meta = SimpleNamespace(obj_user_meta_list=_Node(user_meta))

    embedding = extract_reid_embedding(
        obj_meta,
        pyds_module=_pyds(_layer(**layer_overrides)),
        buffer_reader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("incompatible buffer must not be read")
        ),
    )

    assert embedding is None


def test_frame_embedding_map_is_keyed_by_unchanged_tracker_object_id():
    tensor_meta = SimpleNamespace(unique_id=2, num_output_layers=1)
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(meta_type=99),
        user_meta_data=tensor_meta,
    )
    object_meta = SimpleNamespace(
        object_id=17,
        obj_user_meta_list=_Node(user_meta),
    )
    frame_meta = SimpleNamespace(obj_meta_list=_Node(object_meta))
    pyds = _pyds(_layer(inferDims=SimpleNamespace(numElements=2)))
    pyds.NvDsObjectMeta = _Cast

    embeddings = extract_frame_reid_embeddings(
        frame_meta,
        pyds_module=pyds,
        feature_size=2,
        buffer_reader=lambda _layer, _size: np.array([3.0, 4.0], dtype=np.float32),
    )

    assert set(embeddings) == {17}
    np.testing.assert_allclose(embeddings[17], [0.6, 0.8])
