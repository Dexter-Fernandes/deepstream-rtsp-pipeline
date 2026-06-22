import numpy as np

from pipelines.frame_accessor import get_frame_rgba, write_frame_rgba


def _fake_surface(h=480, w=640):
    return np.zeros((h, w, 4), dtype=np.uint8)


def test_get_frame_rgba_returns_none_when_surface_is_none():
    result = get_frame_rgba(object(), 0, _get_surface=lambda _buf, _idx: None)
    assert result is None


def test_get_frame_rgba_returns_ndarray():
    result = get_frame_rgba(object(), 0, _get_surface=lambda _buf, _idx: _fake_surface())
    assert isinstance(result, np.ndarray)


def test_get_frame_rgba_shape_matches_surface():
    surface = _fake_surface(h=480, w=640)
    result = get_frame_rgba(object(), 0, _get_surface=lambda _buf, _idx: surface)
    assert result.shape == (480, 640, 4)


def test_write_frame_rgba_copies_array_to_surface():
    surface = _fake_surface()
    new_data = np.full((480, 640, 4), 128, dtype=np.uint8)
    write_frame_rgba(surface, new_data)
    assert np.array_equal(surface, new_data)
