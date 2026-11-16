# Push-T rollout gallery

Static gallery for all 50 BD-χ² Push-T test trials, highlighting peak coverage
with mean and median summaries and final coverage for each trial. Posters load
as needed; videos load only after selecting **Play trial**. The page supports
seed search and native video controls.

## Build locally

From the repository root, use Python 3.10 and an installed FFmpeg (including
`ffprobe`):

```bash
python -m pip install numpy==1.26.4 matplotlib==3.9.2 shapely==1.8.4 pillow
python -m zipfile -e results/paper.zip .
python scripts/plots/pusht_gallery.py --root results/paper --output website/build --workers 2
cp website/index.html website/styles.css website/gallery.js website/build/
touch website/build/.nojekyll
python -m http.server 8000 --directory website/build
```

Open <http://localhost:8000>. Serve the build over HTTP; opening `index.html`
directly from the filesystem will block the JSON request in most browsers.

The renderer reads saved trajectories and results; it does not run a policy or a
simulation. It writes `data/trials.json`, 50 MP4 videos, and 50 JPEG posters under
`website/build/`. Generated assets are build output and need not be committed.
All asset URLs are relative so the site works under a GitHub project subpath.

## Published submission gallery

The original 50 trials are preserved at:
<https://roboticexplorationlab.github.io/behavioral-denoising/l4dc/>.

The `main` branch deploys this frozen gallery alongside the current gallery.
Its archived page sources, trial data and media are distributed in the
[`l4dc-submission` release](https://github.com/RoboticExplorationLab/behavioral-denoising/releases/tag/l4dc-submission).
The tag retains the exact submission source before these gallery-link updates.

The workflow on this branch does not deploy, including manual dispatch, so it
cannot overwrite the current gallery. The local build commands above continue
to reproduce the original trial videos.
