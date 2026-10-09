import numpy as np
import pytest

from stackalign import RegisterModel, TransformModel
from stackalign.planes import fit_planes


@pytest.mark.parametrize('backend', ['scikit', 'pystackreg', 'cv2'])
@pytest.mark.parametrize('mode', ['time', 'channel'])
def test_restored_model_plane_matches_whole_stack(backend, mode, monkeypatch):
    monkeypatch.setattr('stackalign.backends.execution.EXECUTOR_MODE', 'thread')
    image = np.random.default_rng(13).integers(0, 1000, (2, 2, 24, 25), dtype=np.uint16)
    matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    matrices[1, :2, 2] = (1.25, -2.5)
    model = TransformModel(mode, 'translation', matrices, 0 if mode == 'channel' else None)
    register = RegisterModel(backend).set_model(model)
    detached = register.model
    detached.transform[:] = 0
    assert register.model.transform[0, 0, 0] == 1
    whole = register.apply(image, 'TCYX')
    for frame in range(2):
        for channel in range(2):
            plane = register.apply_plane(image[frame, channel], frame=frame, channel=channel)
            np.testing.assert_array_equal(plane, whole[frame, channel])


@pytest.mark.parametrize('backend', ['scikit', 'pystackreg', 'cv2'])
@pytest.mark.parametrize('reference', ['previous', 'first', 'mean'])
def test_serial_fit_matches_backend_time_fitting(backend, reference, monkeypatch):
    from scipy.ndimage import gaussian_filter
    monkeypatch.setattr('stackalign.backends.execution.EXECUTOR_MODE', 'thread')
    first = gaussian_filter(np.random.default_rng(3).random((48, 48)), 2).astype(np.float32)
    array = np.stack([first, np.roll(first, 1, axis=0), np.roll(first, 2, axis=0)])
    expected = RegisterModel(backend).fit_time(array, 'TYX', reference_strategy=reference).model
    actual = fit_planes(array, backend=backend, method='translation', reference=reference)
    np.testing.assert_allclose(actual, expected.transform, atol=1e-6)


def test_serial_fit_can_cancel():
    with pytest.raises(InterruptedError):
        fit_planes(np.ones((3, 8, 8)), backend='scikit', method='translation',
                   reference='previous', cancelled=lambda: True)
