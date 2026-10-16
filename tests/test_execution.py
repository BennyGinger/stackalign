from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from time import sleep
from typing import cast

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from stackalign import ExecutionMode, RegisterModel, TransformModel
from stackalign.backends import execution


@pytest.fixture
def movie():
    """Build a small textured movie with known drift and three related channels."""
    base = gaussian_filter(np.random.default_rng(9).random((48, 56)), 1.5).astype(np.float32)
    frames = np.stack([np.roll(base, (frame, -frame), axis=(0, 1)) for frame in range(4)])
    return np.stack([frames, frames * 2, frames * 3], axis=1)


@pytest.mark.parametrize("backend,method", [
    ("scikit", "translation"), ("pystackreg", "translation"),
    ("pystackreg", "rigid_body"), ("pystackreg", "affine"),
    ("cv2", "translation"), ("cv2", "rigid_body"), ("cv2", "affine"),
])
@pytest.mark.parametrize("reference", ["first", "mean", "previous"])
@pytest.mark.parametrize("mode", ["auto", "thread", "process"])
def test_parallel_time_fit_and_apply_match_serial(movie, backend, method, reference, mode):
    """Check all backend methods and reference strategies against serial execution."""
    baseline = RegisterModel(backend, execution="serial").fit_time(
        movie, "TCYX", method=method, reference_strategy=reference, fit_channel=0)
    progress = []
    caller_thread = get_ident()
    parallel = RegisterModel(backend, execution=mode, workers=2).fit_time(
        movie, "TCYX", method=method, reference_strategy=reference, fit_channel=0,
        progress=lambda done, total: progress.append((done, total, get_ident())))
    np.testing.assert_allclose(parallel.model.transform, baseline.model.transform, atol=1e-6)
    np.testing.assert_allclose(parallel.apply(movie, "TCYX"), baseline.apply(movie, "TCYX"), atol=1e-5)
    total = len(movie) - (reference == "previous")
    assert [(done, count) for done, count, _ in progress] == [(i, total) for i in range(total + 1)]
    assert all(thread == caller_thread for _, _, thread in progress)


@pytest.mark.parametrize("backend", ["scikit", "pystackreg", "cv2"])
@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_prepared_channel_planes_exclusions_and_output_memmap(movie, backend, mode, tmp_path):
    """Check prepared channel fits, exclusions, and direct mapped output in every mode."""
    # Add different channel shifts and keep an excluded channel distinct.
    movie[:, 1] = np.roll(movie[:, 1], 1, axis=-1)
    movie[:, 2] += 10
    source = np.asarray(movie * 1000, dtype=np.uint16)
    register = RegisterModel(backend, execution=mode, workers=2).fit_channel_planes(
        source[1], reference_channel=0, exclude_channels=[2])
    serial = RegisterModel(backend, execution="serial").fit_channel(
        source, "TCYX", reference_frame=1, reference_channel=0, exclude_channels=[2])
    np.testing.assert_allclose(register.model.transform, serial.model.transform, atol=1e-6)
    output = np.lib.format.open_memmap(tmp_path / "output.npy", mode="w+",
                                       dtype=source.dtype, shape=source.shape)
    assert register.apply(source, "TCYX", out=output) is output
    np.testing.assert_array_equal(output, serial.apply(source, "TCYX"))
    np.testing.assert_array_equal(output[:, 0], source[:, 0])
    np.testing.assert_array_equal(output[:, 2], source[:, 2])


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_prepared_time_planes_and_permuted_axes(movie, mode, tmp_path):
    """Check prepared mapped inputs and application with noncanonical axes."""
    source = np.lib.format.open_memmap(tmp_path / "planes.npy", mode="w+",
                                       dtype=np.float32, shape=movie[:, 0].shape)
    source[:] = movie[:, 0]
    register = RegisterModel("scikit", execution=mode, workers=2).fit_time_planes(
        source, reference_strategy="previous")
    baseline = RegisterModel("scikit", execution="serial").fit_time(
        movie, "TCYX", fit_channel=0, reference_strategy="previous")
    np.testing.assert_array_equal(register.model.transform, baseline.model.transform)
    expanded = np.stack([movie, movie * 2], axis=2)
    permuted = expanded.transpose(4, 0, 1, 2, 3)
    result = register.apply(permuted, "XTCZY")
    np.testing.assert_array_equal(result, baseline.apply(expanded, "TCZYX").transpose(4, 0, 1, 2, 3))


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_cancellation_keeps_previous_fitted_model(movie, mode):
    """Ensure cancelling a refit preserves the previously accepted matrix set."""
    register = RegisterModel("scikit", execution=mode, workers=2).fit_time(
        movie, "TCYX", fit_channel=0)
    before = register.model
    completed = []
    with pytest.raises(InterruptedError):
        register.fit_time_planes(movie[:, 0], reference_strategy="previous",
                                 progress=lambda done, total: completed.append(done),
                                 cancelled=lambda: bool(completed and completed[-1] >= 1))
    np.testing.assert_array_equal(register.model.transform, before.transform)
    with pytest.raises(InterruptedError):
        register.apply(movie, "TCYX", cancelled=lambda: True)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_workers_rejected(workers):
    """Reject nonpositive, boolean, and fractional worker counts."""
    with pytest.raises(ValueError, match="workers"):
        RegisterModel(workers=workers)


def test_invalid_execution_and_output_rejected(movie):
    """Reject unsupported execution modes and incompatible or overlapping outputs."""
    with pytest.raises(ValueError, match="execution"):
        RegisterModel(execution=cast(ExecutionMode, "invalid"))
    register = RegisterModel("scikit", execution="serial").fit_time(movie, "TCYX", fit_channel=0)
    with pytest.raises(ValueError, match="overlap"):
        register.apply(movie, "TCYX", out=movie)
    with pytest.raises(ValueError, match="shape and dtype"):
        register.apply(movie, "TCYX", out=np.empty(movie.shape, np.uint16))


def test_one_pool_for_all_channels_and_z_planes(movie, monkeypatch):
    """Ensure whole-stack application reuses one pool across channels and Z planes."""
    pools = []

    def create_pool(mode, workers):
        """Record pool creation while retaining real concurrent execution."""
        pools.append((mode, workers))
        return ThreadPoolExecutor(max_workers=workers)

    monkeypatch.setattr(execution, "create_executor", create_pool)
    model = TransformModel("time", "translation", np.repeat(np.eye(3)[None], len(movie), axis=0))
    register = RegisterModel("scikit", execution="thread", workers=2).set_model(model)
    expanded = np.stack([movie, movie * 2], axis=2)
    output = register.apply(expanded, "TCZYX")
    assert output.shape == expanded.shape
    assert pools == [("thread", 2)]


def slow_task(index):
    """Delay one indexed result to exercise bounded scheduling and cancellation."""
    sleep(0.03)
    return index, np.array([index])


def test_queue_is_bounded_and_cancellation_stops_submissions():
    """Ensure cancellation stops a bounded queue before the whole workload is submitted."""
    submitted = []
    completed = []

    def tasks():
        """Record each task requested by the scheduler."""
        for index in range(100):
            submitted.append(index)
            yield (index,)

    with pytest.raises(InterruptedError):
        list(execution.run_tasks(slow_task, tasks(), total=100, execution="thread", workers=2,
                                 progress=lambda done, total: completed.append(done),
                                 cancelled=lambda: bool(completed and completed[-1] >= 1)))
    assert len(submitted) <= 4


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_empty_work_does_not_start_pool(mode, monkeypatch):
    """Ensure reference and excluded channels do not start an unnecessary pool."""
    monkeypatch.setattr(execution, "create_executor", lambda *args: pytest.fail("Unneeded pool"))
    array = np.ones((2, 8, 9), dtype=np.uint16)
    register = RegisterModel("scikit", execution=mode).fit_channel_planes(
        array, reference_channel=0, exclude_channels=[1])
    np.testing.assert_array_equal(register.model.transform, np.repeat(np.eye(3)[None], 2, axis=0))


def test_process_execution_from_background_thread(movie):
    """Exercise spawned processes from a caller thread as used by GUI workers."""
    def fit_and_apply():
        """Fit and apply from a background caller rather than the main thread."""
        return RegisterModel("scikit", execution="process", workers=2).fit_time(
            movie, "TCYX", fit_channel=0, reference_strategy="previous").apply(movie, "TCYX")

    with ThreadPoolExecutor(max_workers=1) as caller:
        actual = caller.submit(fit_and_apply).result(timeout=30)
    expected = RegisterModel("scikit", execution="serial").fit_time(
        movie, "TCYX", fit_channel=0, reference_strategy="previous").apply(movie, "TCYX")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_cancelled_application_leaves_only_partial_caller_output(movie, mode):
    """Ensure interrupted application raises instead of publishing incomplete output."""
    register = RegisterModel("scikit", execution=mode, workers=2).fit_time(
        movie, "TCYX", fit_channel=0)
    output = np.full_like(movie, np.nan)
    completed = []
    with pytest.raises(InterruptedError):
        register.apply(movie, "TCYX", out=output,
                       progress=lambda done, total: completed.append(done),
                       cancelled=lambda: bool(completed and completed[-1] >= 1))
    assert np.isfinite(output).any()
    assert np.isnan(output).any()


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_backend_failure_preserves_previous_model(mode):
    """Propagate native fitting errors without replacing a previous fitted model."""
    matrices = np.repeat(np.eye(3)[None], 3, axis=0)
    register = RegisterModel("cv2", execution=mode, workers=2).set_model(
        TransformModel("time", "translation", matrices))
    with pytest.raises(RuntimeError, match="ECC fit failed"):
        register.fit_time_planes(np.zeros((3, 24, 24), np.float32))
    np.testing.assert_array_equal(register.model.transform, matrices)


@pytest.mark.parametrize("backend,method,operation,pixels,jobs,expected", [
    ("pystackreg", "affine", "fit", 512**2, 3, "process"),
    ("pystackreg", "translation", "fit", 512**2, 3, "serial"),
    ("pystackreg", "rigid_body", "fit", 1024**2, 3, "process"),
    ("pystackreg", "translation", "apply", 512**2, 16, "process"),
    ("pystackreg", "affine", "fit", 256**2, 3, "serial"),
    ("pystackreg", "translation", "fit", 256**2, 100, "process"),
    ("cv2", "affine", "fit", 256**2, 3, "thread"),
    ("cv2", "translation", "apply", 128**2, 16, "serial"),
    ("scikit", "translation", "apply", 128**2, 16, "thread"),
    ("scikit", "translation", "fit", 64**2, 1000, "serial"),
    ("pystackreg", "affine", "fit", 2048**2, 1, "serial"),
    (None, "translation", "fit", 2048**2, 100, "serial"),
])
def test_auto_policy_uses_backend_operation_size_and_jobs(monkeypatch, backend, method, operation,
                                                         pixels, jobs, expected):
    """Check representative benchmark sizes, many small jobs, and single-job work."""
    monkeypatch.setattr(execution, "available_cpus", lambda: 4)
    monkeypatch.setattr(execution, "available_memory", lambda: 16 * 1024**3)
    assert execution.select_execution("auto", backend=backend, method=method, operation=operation,
                                      plane_pixels=pixels, total=jobs) == expected


@pytest.mark.parametrize("mode", ["serial", "thread", "process"])
def test_explicit_modes_override_automatic_resource_guards(monkeypatch, mode):
    """Keep explicit overrides even when automatic selection would run serially."""
    monkeypatch.setattr(execution, "available_cpus", lambda: 1)
    monkeypatch.setattr(execution, "available_memory", lambda: 1)
    assert execution.select_execution(mode, backend="pystackreg", method="affine", operation="fit",
                                      plane_pixels=1024**2, total=4, workers=1) == mode


def test_auto_respects_worker_cpu_and_memory_limits(monkeypatch):
    """Avoid auto pools with one usable CPU/worker or insufficient process workspace."""
    def choose(workers: int | None = None) -> ExecutionMode:
        """Resolve a substantial workload with the currently mocked resources."""
        return execution.select_execution("auto", backend="pystackreg", method="affine",
                                          operation="fit", plane_pixels=1024**2, total=20,
                                          workers=workers)

    monkeypatch.setattr(execution, "available_cpus", lambda: 4)
    monkeypatch.setattr(execution, "available_memory", lambda: 16 * 1024**3)
    assert choose(workers=1) == "serial"
    monkeypatch.setattr(execution, "available_cpus", lambda: 1)
    assert choose() == "serial"
    monkeypatch.setattr(execution, "available_cpus", lambda: 4)
    monkeypatch.setattr(execution, "available_memory", lambda: 128 * 1024**2)
    assert choose() == "serial"
    monkeypatch.setattr(execution, "available_memory", lambda: None)
    assert choose() == "process"


def test_auto_counts_real_jobs_for_channel_exclusions_and_permuted_axes(movie, monkeypatch):
    """Feed the policy only fitted channels and transformed planes, including Z."""
    observed = []

    def record_selection(mode, **options):
        """Capture workload metadata without starting pools for this small fixture."""
        observed.append(options)
        return "serial"

    monkeypatch.setattr(execution, "select_execution", record_selection)
    register = RegisterModel("scikit").fit_channel_planes(movie[0], reference_channel=0,
                                                         exclude_channels=[2])
    assert register.execution == "auto"
    expanded = np.stack([movie, movie * 2], axis=2)
    register.apply(expanded.transpose(4, 0, 1, 2, 3), "XTCZY")
    assert [(row["operation"], row["total"], row["plane_pixels"]) for row in observed] == [
        ("fit", 1, 48 * 56), ("apply", 4 * 2 * 2, 48 * 56)]


@pytest.mark.parametrize("backend,method,side", [("scikit", "translation", 256),
                                                 ("pystackreg", "affine", 512)])
def test_auto_parallel_paths_match_serial(backend, method, side, monkeypatch):
    """Exercise automatic thread/process fits and application against real serial kernels."""
    monkeypatch.setattr(execution, "available_cpus", lambda: 4)
    monkeypatch.setattr(execution, "available_memory", lambda: 16 * 1024**3)
    base = gaussian_filter(np.random.default_rng(7).random((side, side)), 2).astype(np.float32)
    array = np.stack([np.roll(base, (index, -index), axis=(0, 1)) for index in range(4)])
    pools = []
    create_executor = execution.create_executor

    def record_pool(mode, workers):
        """Record the selected mode while exercising the real executor."""
        pools.append(mode)
        return create_executor(mode, workers)

    monkeypatch.setattr(execution, "create_executor", record_pool)
    serial = RegisterModel(backend, execution="serial").fit_time_planes(
        array, method=method, reference_strategy="previous")
    automatic = RegisterModel(backend).fit_time_planes(
        array, method=method, reference_strategy="previous")
    np.testing.assert_allclose(automatic.model.transform, serial.model.transform, rtol=0, atol=1e-6)
    multichannel = np.broadcast_to(array[:, None], (4, 4, side, side))
    np.testing.assert_array_equal(automatic.apply(multichannel, "TCYX"), serial.apply(multichannel, "TCYX"))
    assert pools == (["process", "process"] if backend == "pystackreg" else ["thread", "thread"])
