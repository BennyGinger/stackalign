# Rough execution cutoff estimate

Measured 2026-10-16 in 72 seconds, using four workers.

Four real LTB4 frames, four channels, uint16; images resized to 128, 256, 512 and 1024 pixels per side. Fit channel 1, previous-frame reference. This gives three fitting jobs and sixteen application jobs per case.

One pass per case, fresh pools, worker startup included. Input reading, resizing, backend warm-up and equality checks are outside the timings. Application writes an in-memory uint16 array; no TIFF writing is timed.

## Practical starting estimates

These combine fitting and application. Use serial below the bracket, then try the indicated mode at its upper end. Sizes are square-image side lengths in pixels, for the four-frame workload described above.

| Backend / method | Parallel mode | Estimated cutoff |
|---|---|---:|
| pystackreg / translation | process | 512–1024 |
| pystackreg / rigid_body | process | 256–512 |
| pystackreg / affine | process | 256–512 |
| cv2 / translation | thread | ≤128 |
| cv2 / rigid_body | thread | 128–256 |
| cv2 / affine | thread | 128–256 |
| scikit / translation | thread | ≤128 |

For tiny cases the difference can be only a few milliseconds; serial remains reasonable even when a percentage gain is reported.

## Separate fitting and application estimates

Ranges are **pixels per side for square images**, bracketing the first tested size where parallel work took at least 20% less time than serial and kept that gain at larger tested sizes. This margin avoids treating tiny timing differences as useful gains. `>1024` means no clear gain within the tested sizes; it is not a measured crossover above that size. `inconsistent` means an isolated gain did not persist; no stable cutoff was estimated.

| Backend / method | Fit: threads | Fit: processes | Apply: threads | Apply: processes |
|---|---:|---:|---:|---:|
| pystackreg / translation | >1024 | 512–1024 | >1024 | 256–512 |
| pystackreg / rigid_body | inconsistent | 512–1024 | >1024 | 256–512 |
| pystackreg / affine | inconsistent | 256–512 | >1024 | 256–512 |
| cv2 / translation | ≤128 | >1024 | inconsistent | >1024 |
| cv2 / rigid_body | 128–256 | >1024 | 128–256 | >1024 |
| cv2 / affine | 128–256 | 512–1024 | inconsistent | >1024 |
| scikit / translation | 128–256 | >1024 | ≤128 | 512–1024 |

## How to use this estimate

- Start with serial below a range; try the indicated parallel mode at its upper end.
- Fit and apply can have different cutoffs. The public execution setting currently selects one mode for both.
- More independent frames can make parallel work worthwhile at smaller image sizes. These ranges describe this four-frame workload, not every stack shape or computer.
- These measurements seed StackAlign's conservative `auto` policy; explicit modes remain available.

## Checks and reproducibility

Successful cases: 84/84. Every successful parallel case was compared with serial for fitted matrices (absolute tolerance 1e-6) and exact output pixels.

Source, read only: `/media/ben/Analysis/Python/Images/Boldi/LTB4/500k-HoxB8-c2023-c2028_zym-slides_1uM-LTB4@30m_001_s1/fits_array.tif`.

OpenCV native threads: 16. Parallel worker limit: 4.

Raw seconds and checks: [timings.csv](timings.csv).

Rerun from the FITS repository with:

```bash
.venv/bin/python stackalign/benchmarks/estimate_cutoff.py "/media/ben/Analysis/Python/Images/Boldi/LTB4/500k-HoxB8-c2023-c2028_zym-slides_1uM-LTB4@30m_001_s1/fits_array.tif"
```
