# stackalign (internal)

Lightweight registration module used inside the workspace.

This is **not a general-purpose library**.
Design favors:

* clarity
* deterministic behavior
* performance on real microscopy stacks

---

## Core idea

Everything is built around **explicit transform matrices**:

* time-wise → `(T, 3, 3)`
* channel-wise → `(C, 3, 3)`

Fit once → apply anywhere (same shape semantics)

No hidden state, no implicit behavior.

---

## Pipeline overview

1. **axes.py**

   * validate + normalize axes (`T, C, Z, Y, X`)
   * reorder to canonical layout

2. **preparation.py**

   * build fit views:

     * time → `TYX`
     * channel → `CYX`
   * collapse Z via max projection (fit only)
   * restore original shape/dtype after apply

3. **backends/pystackreg/**

   * `time_wise.py` → time registration logic
   * `channel_wise.py` → channel registration logic
   * `execution.py` → parallel execution (process/thread)
   * `utils.py` → helpers (StackReg, tmats, accumulation)

4. **models.py**

   * `TransformModel` stores:

     * mode
     * method
     * transform matrices
     * reference_channel (channel mode)

5. **api.py**

   * user entry point (`RegisterModel`)
   * dispatches to backend

---

## Key design decisions

### 1. No stack-native pystackreg

We do **not** use:

* `register_stack`
* `transform_stack`

Reason:
→ too slow on real datasets

Instead:

* explicit per-frame / per-channel transforms
* parallel execution

---

### 2. "previous" is cumulative (important)

We do NOT apply pairwise transforms directly.

Process:

```
pairwise:   t -> t-1
accumulate: t -> frame 0
```

So final result is globally aligned.

See:

* `utils.accumulate_pairwise_tmats`

---

### 3. Parallelism is internal

* default: process-based
* controlled in `execution.py`
* user is not aware of it

No public config. Keep it that way.

---

### 4. Fit is simplified, apply is full

Fit:

* single channel (if needed)
* Z max projection
* float32

Apply:

* full data (T/C/Z preserved)
* dtype restored exactly

---

### 5. Fail fast

We do NOT auto-infer:

* missing `T` for time-wise → error
* missing `C` for channel-wise → error
* missing `reference_channel` → error

No silent behavior.

---

## Performance model

Time-wise:

* fit → parallel frame registration
* apply → parallel per-frame transform

Channel-wise:

* fit → cheap (few channels)
* apply → can be parallelized per image

Critical point:
→ large arrays → avoid copying → small tasks

---

## Constraints

* axes must include `Y` and `X`
* channel-wise requires explicit reference channel
* Z motion is ignored during fit (max projection)

---

## What NOT to change lightly

* explicit tmats representation
* `"previous"` accumulation logic
* separation between `preparation` and backend
* internal parallel execution model

These are core to performance and correctness.

---

## Where to look when debugging

* wrong alignment → `time_wise.py` + accumulation
* wrong shape/dtype → `preparation.py`
* performance issues → `execution.py`
* axis issues → `axes.py`

---

## Tests

Tests cover:

* shape/dtype preservation
* axis roundtrip
* TCZYX behavior
* time + channel consistency

Run them before refactoring anything critical.

---

## Mental model

Think of this module as:

> "build transforms on a simple view → apply them everywhere safely"

Not:

> "smart registration system"

Keep it explicit.
