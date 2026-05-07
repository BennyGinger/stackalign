# stackalign (internal)

Lightweight registration module for microscopy stacks.
**Not a public library.** Designed for clarity, determinism, and real-world performance.

---

## Core idea

Everything is based on **explicit transform matrices**:

* time-wise → `(T, 3, 3)`
* channel-wise → `(C, 3, 3)`

Fit once → apply anywhere
No hidden state, no implicit behavior.

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

   * `time_wise.py` / `channel_wise.py` → logic
   * `execution.py` → parallelism
   * backend-specific implementations
4. **models.py** → `TransformModel`
5. **api.py** → `RegisterModel` entry point

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

* process-based by default
* handled in `execution.py`
* not exposed to user

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
