# stackalign (internal)

Lightweight registration module for microscopy stacks. It estimates explicit
2D transform matrices for time-wise drift or cross-channel offsets, then applies
those transforms to the complete array while preserving its shape and dtype.
OpenCV, scikit-image, and pystackreg implementations share the same façade.

**Not a public library.** Designed for clarity, determinism, and real-world performance.

Within FITS, `stackalign` owns registration mathematics only. FITS resolves the
registration scenario, loads arrays and axes, records provenance, and saves the
registered artifact.

---

## Core idea

Everything is based on **explicit transform matrices**:

* time-wise → `(T, 3, 3)`
* channel-wise → `(C, 3, 3)`

Fit once → apply anywhere
No hidden state, no implicit behavior.

`RegisterModel.model` returns a detached `TransformModel` for persistence;
`RegisterModel(backend).set_model(model)` restores it without fitting.
`apply_plane(image, frame=..., channel=...)` applies the corresponding matrix
with the same backend interpolation and dtype restoration as whole-stack apply.
The backend name must be retained alongside a model because matrix conventions
and interpolation differ between backends. Public `fit_time_planes()` and
`fit_channel_planes()` methods accept prepared TYX/CYX arrays, including memmaps,
with the same execution, progress, cancellation, and exclusion behavior as the
full-array API. Persistence and input/cache validation remain FITS-owned.

```python
from stackalign import RegisterModel

register = RegisterModel("pystackreg", execution="auto", workers=4)
register.fit_time(array, "TCYX", fit_channel=1, method="affine",
                  reference_strategy="previous", progress=on_progress,
                  cancelled=is_cancelled)
registered = register.apply(array, "TCYX", out=output_memmap,
                            progress=on_progress, cancelled=is_cancelled)
```

The four modes are `"auto"` (default), `"serial"`, `"thread"`, and `"process"`.
Auto chooses separately for fitting and application using the backend, method,
pixels per plane, actual independent job count, and available CPUs. It uses
conservative estimates from [the short benchmark](benchmarks/README.md), without
running benchmarks during registration. Small work and single-job operations
run serially; larger PyStackReg work uses processes, while OpenCV/scikit-image
use threads. Explicit modes override this policy.

`workers=None` uses up to four available CPUs, capped by the number of tasks.
Auto also caps an explicit worker limit by CPU availability. Where the OS exposes
available physical memory, auto avoids processes if estimated worker workspaces
would exceed half of that memory. This is a rough guard; native backend memory
use can vary. Serial execution
creates no pool. Process execution uses spawn, so scripts must launch work under
`if __name__ == "__main__":`; calls from a GUI background thread are supported.

Progress callbacks receive `(completed, total)` for actual fits or transformed
planes, beginning at zero. Both callbacks execute in the calling thread and do
not need to be picklable. Cancellation raises `InterruptedError`, stops further
submissions, cancels queued tasks, and waits for native operations already running.
A cancelled or failed refit keeps the previous fitted model.

Application uses one pool across the complete stack, with at most twice the
worker count queued. Dtype restoration happens per plane, avoiding a full
floating-point output copy. An optional `out` array must be writable, have the
input shape/dtype, and not overlap the input. It may be a memmap. Callers own
flushing, cleanup, and publication; failures can leave partial values in `out`.
Single-image `apply_plane()` runs directly without creating a pool.

---

## Pipeline

1. **axes.py** → validate + normalize (`T, C, Z, Y, X`)
2. **preparation.py**

   * build fit views:

     * time → `TYX`
     * channel → `CYX`
   * Z → max projection (fit only)
   * restore shape/dtype on apply
3. **backends/**

   * `time_wise.py` / `channel_wise.py` → backend kernels
   * `execution.py` → automatic selection and bounded serial/thread/process execution
   * facades → compatibility with the shared public implementation
4. **models.py** → `TransformModel`
5. **planes.py** → shared fitting and per-plane stack application
6. **api.py** → `RegisterModel` entry point

---

## Key design decisions

### 1. Explicit transforms only

* all backends return `(N, 3, 3)` matrices
* deterministic + reusable
* no backend-specific state

---

### 2. No stack-native registration

We do NOT use:

* `register_stack`
* `transform_stack`

Reason:
→ too slow on real data

Instead:

* frame/channel-wise operations
* parallel execution

---

### 3. `"previous"` is cumulative (critical)

```text
pairwise:   t → t-1
accumulate: t → frame 0
```

→ ensures global alignment, not drift chaining

---

### 4. Fit ≠ Apply

Fit:

* reduced view (single channel, Z-projected)
* float32

Apply:

* full data (T/C/Z)
* dtype restored exactly

---

### 5. Parallelism is internal

* explicit serial, thread, or process execution
* worker limit configured per `RegisterModel`
* handled in `execution.py`, shared across backends and fitting/application

---

### 6. Fail fast

No implicit behavior:

* missing `T` → error (time-wise)
* missing `C` → error (channel-wise)
* missing `reference_channel` → error

---

## Backends (practical ranking)

### Time-wise

1. **scikit (phase_cross_correlation)**

   * fastest
   * translation only
   * best default for drift correction

2. **pystackreg**

   * robust
   * translation / rigid / affine
   * slightly better than cv2 in difficult cases

3. **cv2 (ECC)**

   * similar to pystackreg
   * slightly less stable depending on reference

---

### Channel-wise

1. **cv2 (ECC)**

   * fastest (fit + apply)
   * supports rigid_body (useful for dual-camera)

2. **scikit**

   * fast
   * translation only

3. **pystackreg**

   * slowest
   * still reliable fallback

---

## Usage rules (important)

### Time-wise

* corrects **sample drift**
* best strategy:

  * `"previous"` → default
  * `"mean"` → stable stacks
  * `"first"` → short stacks only

---

### Channel-wise

* corrects **optical misalignment**
* works only if channels share structure

Good:

* GFP ↔ RFP (same cells)
* nucleus ↔ membrane
* dual-camera misalignment

Bad:

* sparse fluorescence vs unrelated signal

Defaults:

* translation → standard
* rigid_body → dual-camera setups
* affine → advanced / use cautiously

---

## Constraints

* axes must include `Y` and `X`
* Z motion is ignored during fit (max projection)
* channel-wise requires explicit reference channel

---

## Performance model

* avoid large array copies
* operate on small independent tasks
* parallelize per frame/channel

`previous` fits each pair of original adjacent frames independently, then
accumulates matrices in frame order. Fitting can therefore use any execution
mode without changing the accumulation order. Thread speed depends on the
backend's native code; process startup and array transfers can outweigh gains
on small workloads. Benchmark the intended workload before choosing a mode.

A short developer benchmark estimates execution cutoffs using four LTB4 frames
at four image sizes, with a three-minute budget. It includes pool startup and
checks matrices and output pixels against serial execution. The source TIFF is
read only; StackAlign uses the estimates in its conservative automatic policy.

```bash
python stackalign/benchmarks/estimate_cutoff.py /path/to/fits_array.tif
```

See [the rough cutoff estimates](benchmarks/README.md) and
[measured timings](benchmarks/timings.csv). These are starting points for this
workload and computer, rather than universal array-size thresholds. The benchmark
requires `tifffile` and OpenCV, available in the FITS development environment.

---

## What NOT to change lightly

* transform representation `(N, 3, 3)`
* `"previous"` accumulation logic
* separation: preparation ↔ backend
* internal parallel execution

---

## Debug guide

* wrong alignment → backend logic (`time_wise.py`)
* drift issues → `"previous"` accumulation
* wrong shape/dtype → `preparation.py`
* slow → `execution.py`
* axis bugs → `axes.py`

---

## Mental model

> Build transforms on a simple view → apply them everywhere safely

Not:

> a “smart” registration system

Keep it explicit.
