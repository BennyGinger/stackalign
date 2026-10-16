"""Estimate execution cutoffs with four frames at four image sizes, within three minutes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import tifffile

from stackalign import RegisterModel
from stackalign.constants import Method
from stackalign.planes import plane_functions

VARIANTS: list[tuple[str, Method]] = [("pystackreg", "translation"), ("pystackreg", "rigid_body"),
            ("pystackreg", "affine"), ("cv2", "translation"),
            ("cv2", "rigid_body"), ("cv2", "affine"), ("scikit", "translation")]
MODES = ("serial", "thread", "process")
SIZES = (128, 256, 512, 1024)


def estimate_range(rows, backend, method, mode, stage):
    """Bracket a twenty-percent gain that persists at larger tested sizes."""
    measurements = []
    for size in SIZES:
        selected = {row["mode"]: row for row in rows if row["backend"] == backend
                    and row["method"] == method and row["side"] == size and row["status"] == "ok"}
        if "serial" not in selected or mode not in selected:
            continue
        if stage == "total_seconds":
            parallel = selected[mode]["fit_seconds"] + selected[mode]["apply_seconds"]
            serial = selected["serial"]["fit_seconds"] + selected["serial"]["apply_seconds"]
        else:
            parallel, serial = selected[mode][stage], selected["serial"][stage]
        measurements.append((size, parallel <= 0.8 * serial))
    if not measurements:
        return "not measured"
    if not measurements[-1][1]:
        return "inconsistent" if any(won for _, won in measurements) else f">{measurements[-1][0]}"
    index = len(measurements) - 1
    while index > 0 and measurements[index - 1][1]:
        index -= 1
    size = measurements[index][0]
    return f"≤{size}" if index == 0 else f"{measurements[index - 1][0]}–{size}"


def write_report(destination, rows, source, elapsed):
    """Save a concise estimate and the small timing table used to derive it."""
    with (destination / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Rough execution cutoff estimate", "",
             f"Measured 2026-10-16 in {elapsed:.0f} seconds, using four workers.", "",
             "Four real LTB4 frames, four channels, uint16; images resized to 128, 256, "
             "512 and 1024 pixels per side. Fit channel 1, previous-frame reference. "
             "This gives three fitting jobs and sixteen application jobs per case.", "",
             "One pass per case, fresh pools, worker startup included. Input reading, "
             "resizing, backend warm-up and equality checks are outside the timings. "
             "Application writes an in-memory uint16 array; no TIFF writing is timed.", "",
             "## Practical starting estimates", "",
             "These combine fitting and application. Use serial below the bracket, "
             "then try the indicated mode at its upper end. Sizes are square-image "
             "side lengths in pixels, for the four-frame workload described above.", "",
             "| Backend / method | Parallel mode | Estimated cutoff |",
             "|---|---|---:|"]
    for backend, method in VARIANTS:
        mode = "process" if backend == "pystackreg" else "thread"
        cutoff = estimate_range(rows, backend, method, mode, "total_seconds")
        lines.append(f"| {backend} / {method} | {mode} | {cutoff} |")
    lines += ["", "For tiny cases the difference can be only a few milliseconds; "
              "serial remains reasonable even when a percentage gain is reported.", "",
             "## Separate fitting and application estimates", "",
             "Ranges are **pixels per side for square images**, bracketing the first "
             "tested size where parallel work took at least 20% less time than serial "
             "and kept that gain at larger tested sizes. "
             "This margin avoids treating tiny timing differences as useful gains. "
             "`>1024` means no clear gain within the tested sizes; it is not a measured "
             "crossover above that size. `inconsistent` means an isolated gain did "
             "not persist; no stable cutoff was estimated.", "",
             "| Backend / method | Fit: threads | Fit: processes | Apply: threads | Apply: processes |",
             "|---|---:|---:|---:|---:|"]
    for backend, method in VARIANTS:
        values = [estimate_range(rows, backend, method, mode, stage)
                  for stage in ("fit_seconds", "apply_seconds") for mode in ("thread", "process")]
        lines.append(f"| {backend} / {method} | " + " | ".join(values) + " |")
    lines += ["", "## How to use this estimate", "",
              "- Start with serial below a range; try the indicated parallel mode at its upper end.",
              "- Fit and apply can have different cutoffs. The public execution setting currently selects one mode for both.",
              "- More independent frames can make parallel work worthwhile at smaller image sizes. "
              "These ranges describe this four-frame workload, not every stack shape or computer.",
              "- These measurements seed StackAlign's conservative `auto` policy; explicit modes remain available.", "",
              "## Checks and reproducibility", "",
              f"Successful cases: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)}. "
              "Every successful parallel case was compared with serial for fitted matrices "
              "(absolute tolerance 1e-6) and exact output pixels.", "",
              f"Source, read only: `{source}`.", "",
              f"OpenCV native threads: {cv2.getNumThreads()}. Parallel worker limit: 4.", "",
              "Raw seconds and checks: [timings.csv](timings.csv).", "",
              "Rerun from the FITS repository with:", "", "```bash",
              f'.venv/bin/python stackalign/benchmarks/estimate_cutoff.py "{source}"', "```", ""]
    (destination / "README.md").write_text("\n".join(lines))


def run(source, destination):
    """Measure a bounded size sweep and verify mode equivalence without source writes."""
    started = perf_counter()
    with tifffile.TiffFile(source) as tif:
        if tif.series[0].axes != "TCYX" or tif.series[0].shape[1] != 4:
            raise ValueError("Expected a four-channel TCYX TIFF.")
        planes = np.stack([tif.pages[index].asarray() for index in range(16)])
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in SIZES:
        array = np.stack([cv2.resize(plane, (size, size), interpolation=cv2.INTER_AREA)
                          for plane in planes]).reshape(4, 4, size, size)
        for backend, method in VARIANTS:
            plane_functions(backend, method)
            baseline_matrices = baseline_output = None
            for mode in MODES:
                if perf_counter() - started >= 180:
                    write_report(destination, rows, source, perf_counter() - started)
                    print("Three-minute budget reached; saved completed estimates.", flush=True)
                    return
                row: dict[str, Any] = dict(side=size, frames=4, channels=4, backend=backend, method=method,
                                           mode=mode, fit_seconds="", apply_seconds="", status="ok", error="")
                try:
                    register = RegisterModel(backend, execution=mode, workers=4)
                    tick = perf_counter()
                    register.fit_time(array, "TCYX", fit_channel=1, method=method,
                                      reference_strategy="previous")
                    row["fit_seconds"] = perf_counter() - tick
                    output = np.empty_like(array)
                    tick = perf_counter()
                    register.apply(array, "TCYX", out=output)
                    row["apply_seconds"] = perf_counter() - tick
                    if mode == "serial":
                        baseline_matrices = register.model.transform.copy()
                        baseline_output = output
                    else:
                        if baseline_matrices is None or baseline_output is None:
                            raise RuntimeError("Serial baseline unavailable for the numerical check.")
                        np.testing.assert_allclose(register.model.transform, baseline_matrices, rtol=0, atol=1e-6)
                        np.testing.assert_array_equal(output, baseline_output)
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                rows.append(row)
            print(f"{size}px {backend}/{method}: finished", flush=True)
        write_report(destination, rows, source, perf_counter() - started)
    print(f"Finished in {perf_counter() - started:.1f}s: {destination / 'README.md'}", flush=True)


def main():
    """Parse the read-only source path and start the short estimate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    run(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
