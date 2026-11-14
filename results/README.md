# Saved paper results

`paper.zip` contains the saved trials, expert references, timing records, and
summary tables for Cartpole, Rocket Landing, Reacher, and Push-T. Extract it
from the repository root:

```bash
python -m zipfile -e results/paper.zip .
```

The archive creates `results/paper/`. See its `README.md` for the benchmark
tables and the main repository README for plotting and test commands.
Extracted files and generated figures are ignored by Git.

The archive is optional for running experiments. New runs go into a separate
output folder, such as `results/reproduction/`. Tests that require the saved
paper results skip until the archive is extracted.
