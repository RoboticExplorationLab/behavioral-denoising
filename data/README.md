# Demonstrations and evaluation starts

All methods use the same demonstrations for each task. The included data are
ready to use with the configurations in `config/`.

| Task | Demonstrations | Episodes | Samples | Test starts |
|---|---|---:|---:|---:|
| Cartpole | Nonlinear MPC, horizon 100 | 200 | 20,000 | 10 |
| Rocket Landing | Linear MPC, horizon 60 | 58 | 17,705 | 10 |
| Reacher | Nonlinear MPC, horizon 30 | 200 | 40,000 | 10 |
| Push-T | Human demonstrations from Diffusion Policy | 206 | 25,650 | 50 |

Each task directory contains `demonstrations.npz` and `test_starts.json`.
MPC costs for the same evaluation starts are in `expert_evaluations.json`.
The MPC controllers are implemented in `src/controllers/`.

Windows are built within episode boundaries. Reacher demonstrations append the
reference x/y coordinates to the eight state features. Its evaluation starts
specify all five references in order.

Push-T observations contain pusher x/y, block x/y, and the cosine and sine of the
block angle. Actions are absolute pusher target positions in pixels. The dataset
omits terminal observations, so alignment drops the last action of each episode,
leaving 25,444 action/next-observation pairs.

## Downloading Push-T data

To download and convert the dataset again, choose a new output directory:

```bash
.venv-pusht/bin/python scripts/prepare_data.py --pusht-output results/pusht_data
```

Use `--archive /path/to/pusht.zip` to read an existing download.
