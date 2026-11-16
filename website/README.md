# Push-T rollout gallery

The gallery shows all 50 BD-χ² trials with block-relative observations and action
targets. Cards show peak and final coverage; videos load only when selected.

## Preview the published gallery

The videos, posters, and coverage summaries are stored as a versioned GitHub
Release asset rather than experiment outputs in the source tree. Python 3.10+
can download the checksummed asset and assemble the site:

```bash
python website/build.py
python -m http.server 8000 --directory website/build
```

Open <http://localhost:8000>. To use an already downloaded archive, pass
`--archive /path/to/pusht-gallery.zip`. The builder refuses to overwrite an
existing directory; use `--output /path/to/new-build` for another build.

## Render your own runs

After running Push-T, use the Push-T environment with FFmpeg installed:

```bash
.venv-pusht/bin/python scripts/plots/pusht_gallery.py \
  --root runs/benchmark --output website/my-build --workers 6
cp website/index.html website/styles.css website/gallery.js website/my-build/
python -m http.server 8000 --directory website/my-build
```

The renderer accepts any completed BD-χ² subset, checks saved records, and draws
recorded poses at native 10 Hz without running a simulator. Render checksums and
video frame counts allow interrupted builds to resume. An optional
`--preview path/to/preview.png` writes a contact sheet.

For per-episode state/control plots and motion, use `scripts/plot.py --videos`.
For coverage summaries alone, use `scripts/plots/pusht.py`. Both accept
`--root runs/benchmark` and work with one or both methods.

## Deployment

GitHub Pages builds from `website/gallery_asset.json` on changes to the website
or workflow on `main`, or through **Actions → Deploy rollout gallery → Run
workflow**. Publish the specified release asset before pushing a new asset
manifest. The workflow verifies its SHA-256 and builds the static site without
installing a simulator or running experiments.

The deployed page is
<https://roboticexplorationlab.github.io/behavioral-denoising/>.
Original paper results and figure reproduction remain on the
[`l4dc` branch](https://github.com/RoboticExplorationLab/behavioral-denoising/tree/l4dc).
