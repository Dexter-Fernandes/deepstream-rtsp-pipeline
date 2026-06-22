import numpy as np


def get_frame_rgba(gst_buffer, frame_idx: int = 0, *, _get_surface=None) -> np.ndarray | None:
    """Map NVMM surface to a host-accessible RGBA numpy array. Returns None on failure.

    _get_surface is injectable for unit tests; production code uses pyds.get_nvds_buf_surface.
    """
    if _get_surface is None:
        import pyds
        _get_surface = pyds.get_nvds_buf_surface

    surface = _get_surface(hash(gst_buffer), frame_idx)
    if surface is None:
        return None
    return np.array(surface, copy=True)


def write_frame_rgba(surface, array: np.ndarray) -> None:
    """Write a modified RGBA numpy array back to the NVMM surface in-place."""
    np.copyto(surface, array)
