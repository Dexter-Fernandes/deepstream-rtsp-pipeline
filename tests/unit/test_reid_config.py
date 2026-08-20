from configparser import ConfigParser
from pathlib import Path

CONFIG = Path(__file__).parents[2] / "configs" / "nvinfer_reid.txt"


def test_reid_config_is_a_sparse_secondary_tensor_output():
    parser = ConfigParser()
    assert parser.read(CONFIG) == [str(CONFIG)]

    properties = parser["property"]
    assert properties.getint("process-mode") == 2
    assert properties.getint("operate-on-gie-id") == 1
    assert properties.getint("operate-on-class-ids") == 0
    assert properties.getint("output-tensor-meta") == 1
    # Gst-nvinfer only consults secondary-reinfer-interval for detector and
    # classifier network types.  Treating the feature vector as "other" (100)
    # would infer every object on every frame.
    assert properties.getint("network-type") == 1
    assert properties.getint("classifier-async-mode") == 0
    # TAO ReIdentificationNet was trained with a direct 256x128 resize; do not
    # letterbox person crops and shift the input distribution.
    assert properties.getint("maintain-aspect-ratio") == 0
    assert properties.getint("symmetric-padding", fallback=0) == 0
    # Generic ``interval`` only skips full-frame (primary) inference.  SGIE
    # object crops are cached and scheduled by this separate property.
    assert properties.getint("secondary-reinfer-interval") >= 1
    assert properties["output-blob-names"] == "fc_pred"
    assert properties["infer-dims"] == "3;256;128"


def test_reid_config_skips_crops_too_small_for_useful_appearance():
    parser = ConfigParser()
    parser.read(CONFIG)

    properties = parser["property"]
    assert properties.getint("input-object-min-width") >= 16
    assert properties.getint("input-object-min-height") >= 32
