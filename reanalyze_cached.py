"""Re-run the analysis over every mask in data/ using cached click points.

main.py's batch block (and pipeline.py) skip any mask that already has a
``_qc.png`` *before* collecting its metrics, so a re-run only accumulates the
masks processed that run -- and ``export_all_metrics_xlsx`` overwrites the file
with just those rows. After every case has a report, the Excel ends up holding a
single (last-processed) row.

This script rebuilds the full ``data/metrics.xlsx`` (and ``metrics.csv``) non-interactively: it analyses
every ``data/*.nii`` that has a cached ``{case}_clicks.json`` (so ``get_user_path``
loads the saved path instead of opening napari), exports all rows at once, and
bundles every ``{case}_qc.png`` into ``data/qc_report.pdf``.
Masks without cached clicks (e.g. the stray ``seg.nii``) are skipped, since they'd
require an interactive path pick.

Usage:
    uv run python reanalyze_cached.py
"""

import glob
import os

import matplotlib

# Headless: no plot windows during a batch run. Must be set before main imports pyplot.
matplotlib.use("Agg")

import main as analysis  # noqa: E402


def main():
    data_dir = "data"
    nii_files = sorted(
        glob.glob(os.path.join(data_dir, "*.nii"))
        + glob.glob(os.path.join(data_dir, "*.nii.gz")),
        key=analysis.natural_key,
    )

    all_metrics = []
    skipped = []
    for nii_path in nii_files:
        clicks_path = analysis.get_clicks_path(nii_path)
        if not os.path.exists(clicks_path):
            skipped.append(nii_path)
            print(f"Skipping {nii_path} (no cached clicks: {os.path.basename(clicks_path)})")
            continue

        print(f"\nProcessing {nii_path} (cached clicks)...")
        metrics = analysis.analyze_vein(nii_path)
        if metrics is not None:
            all_metrics.append((nii_path, metrics))
        else:
            print(f"  No metrics produced for {nii_path}.")

    if all_metrics:
        analysis.export_all_metrics(all_metrics, data_dir)
        analysis.export_qc_pdf(data_dir)
        print(f"\nDone. {len(all_metrics)} case(s) written to {data_dir}/metrics.xlsx/.csv.")
    else:
        print("\nNo metrics produced.")

    if skipped:
        print(f"Skipped {len(skipped)} mask(s) without cached clicks: "
              f"{', '.join(os.path.basename(p) for p in skipped)}")


if __name__ == "__main__":
    main()
