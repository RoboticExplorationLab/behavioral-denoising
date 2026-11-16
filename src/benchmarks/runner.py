"""Configured benchmarks with atomic per-start records and checked resumption."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from importlib import import_module, metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from src.data import prepare_dataset
from src.dataset_io import load_dataset
from src.evaluation import load_experiment_config, load_method_configs
from .records import _atomic_write, _write_json, export_episode
from .recording import TraceRecorder


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("cartpole", "reacher", "rocket", "pusht")


def json_value(value):
    """Strict JSON for manifests and diagnostics; numerical arrays stay in NPZ."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": "NaN" if np.isnan(value) else "+Infinity" if value > 0 else "-Infinity"}
    return value


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def starts_file(task, configuration):
    return configuration.get("test_starts", configuration.get("test_starts_file", f"data/{task}/test_starts.json"))


def canonical_method(identifier):
    return identifier.removesuffix("_upstream")


def atomic_json_lines(path, records):
    content = "".join(json.dumps(json_value(record), allow_nan=False) + "\n" for record in records)
    _atomic_write(Path(path), lambda stream: stream.write(content.encode()))


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def checked_input(path):
    path = resolve(path)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError:
        raise ValueError("Benchmark inputs must be inside the data directory") from None
    expected = manifest.get(relative)
    actual = digest(path)
    if expected is None or expected["sha256"] != actual:
        raise ValueError(f"Input is absent from, or differs from, data/manifest.json: {relative}")
    return path, actual


def protocol_manifest(task, configuration, methods, starts, inputs):
    """Identity includes parameters, inputs, source and numerical dependencies."""
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "src/tasks/assets").glob("*.xml"))
    sources += [ROOT / "experiments" / f"{task}.py"]
    if task == "pusht":
        sources += list((ROOT / "third_party/diffusion_policy").rglob("*.py"))
    if task == "rocket":
        sources += list((ROOT / "third_party/rocket_lander/coco_rocket_lander").rglob("*.py"))
    packages = {}
    for package in ("numpy", "scipy", "cvxpy", "osqp", "scikit-learn", "gymnasium", "mujoco",
                    "casadi", "coco-rocket-lander", "pymunk", "shapely", "gym"):
        try:
            packages[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            pass
    protocol = json_value({
        "schema_version": 1, "task": task, "configuration": configuration, "methods": methods,
        "test_starts": starts, "input_sha256": inputs,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(sources))},
        "python": platform.python_version(), "packages": packages,
    })
    encoded = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    protocol["sha256"] = hashlib.sha256(encoded).hexdigest()
    return protocol


def initialize_output(output, task, protocol):
    path = output / "protocols" / f"{task}.json"
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError(f"Changed protocol/source/input; use a fresh output folder: {output}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, protocol)


def completed_record(directory, protocol_sha256):
    """A completion marker is valid only with all of its checksummed payloads."""
    if not (directory / "result.json").exists():
        return False
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Completed record belongs to another protocol: {directory}")
    checksums = metadata.get("payload_sha256", {})
    if "trajectory.npz" not in checksums or "result.json" not in checksums:
        raise ValueError(f"Completed record has no payload checksums: {directory}")
    for relative, expected in checksums.items():
        path = directory / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Incomplete or changed completed record: {path}")
    return True


def publish_record(staging, destination, metadata):
    """Publish result.json last, so an interrupted episode is never skipped."""
    metadata["payload_sha256"] = {str(path.relative_to(staging)): digest(path)
                                  for path in staging.rglob("*") if path.is_file() and path.name != "metadata.json"}
    metadata["trajectory_sha256"] = digest(staging / "trajectory.npz")
    _write_json(staging / "metadata.json", metadata)
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "result.json")
    for source in files + [staging / "result.json"]:
        target = destination / source.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)


def experiment_starts(task, module, configuration):
    """Load the evaluation seeds, initial conditions, and reference sequences."""
    path, _ = checked_input(starts_file(task, configuration))
    saved = json.loads(path.read_text())
    if saved["task"] != task:
        raise ValueError("Test-start manifest has the wrong task")
    if task == "cartpole":
        steps = configuration.get("steps", saved["max_steps"])
    elif task == "pusht":
        steps = configuration.get("task", {}).get("max_episode_steps", saved["max_steps"])
    else:
        steps = configuration.get("evaluation", {}).get("steps", saved["max_steps"])
    if type(steps) is not int or steps < 1:
        raise ValueError("The configured horizon must be a positive integer")
    starts = []
    for record in saved["starts"]:
        initial = record.get("targets") if task == "reacher" else record.get("initial_condition")
        starts.append((int(record["seed"]), initial, steps))
    return starts


def saved_experts(task, configuration, starts):
    configured = configuration.get("evaluation", {}).get("expert_cache_file")
    configured = configured or configuration.get("expert_cache_file", f"data/{task}/expert_evaluations.json")
    path, checksum = checked_input(configured)
    saved = json.loads(path.read_text())
    by_seed = {}
    for record in saved["runs"]:
        seed = int(record["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate expert seed: {seed}")
        by_seed[seed] = record
    experts = {}
    frozen = json.loads(resolve(starts_file(task, configuration)).read_text())
    frozen_starts = {record["seed"]: record for record in frozen["starts"]}
    for seed, start, _ in starts:
        if seed not in by_seed:
            raise ValueError(f"No archived expert for seed {seed}")
        record = by_seed[seed]
        key = "targets" if task == "reacher" else "start"
        archived_argument = frozen_starts[seed].get("target_argument", start) if task == "reacher" else start
        if key in record and record[key] != json_value(archived_argument):
            raise ValueError(f"Archived expert start differs for seed {seed}")
        result = deepcopy(record["result"])
        result["seed"] = seed
        experts[seed] = result
    return experts, {str(path.relative_to(ROOT)): checksum}


def expert_comparison_matches(task, configuration):
    """Only compare saved expert costs under the same evaluation conditions."""
    configured = configuration.get("evaluation", {}).get("expert_cache_file")
    configured = configured or configuration.get("expert_cache_file", f"data/{task}/expert_evaluations.json")
    path, _ = checked_input(configured)
    reference = json.loads(path.read_text()).get("evaluation_conditions")
    if reference is None:
        return False
    current = deepcopy(configuration)
    if task == "cartpole":
        current.setdefault("state_weights", [5, 0, 10000, .1, .1])
        current.setdefault("action_weights", [1])
    return all(current.get(key) == value for key, value in reference.items())


def run_controller_trial(task, module, configuration, method, parameters, data, start, cache):
    seed, initial, steps = start
    recorder = TraceRecorder(task)
    if task == "cartpole":
        policy = module.build_policy(cache, method, parameters)
        row = module.run_episode(policy, seed, configuration["task"], steps, configuration["reference"],
                                 True, configuration["outcome_protocol"], recorder=recorder)
    elif task == "reacher":
        is_deepc = method["family"] in module.DEEPC_FAMILIES
        policy_data = module.strip_targets(data) if is_deepc else data
        policy = module.build_policy(method, parameters, policy_data, configuration, cache=cache[is_deepc])
        row = module.episode(policy.act, policy.reset, seed, initial, steps, not is_deepc,
                             trace=True, settings=configuration, recorder=recorder)
    else:
        make = lambda environment: module.build_policy(method, parameters, data, environment, cache)
        row = module.episode(make, seed, initial, steps, trace=True, settings=configuration, recorder=recorder)
    row["seed"] = seed
    return row, recorder


def run_task(task, *, config_path=None, methods=None, seeds=None, steps=None, output=None, overrides=None,
             representation=None):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    module = import_module(f"experiments.{task}")
    configuration = load_experiment_config(task, config_path)
    if overrides:
        configuration.update(deepcopy(overrides))
    if configuration.get("evaluation_mode") != "fixed":
        raise ValueError("Release benchmarks use fixed parameters, not a tuning grid")
    method_configs = load_method_configs(task, configuration["methods"])
    if representation is not None:
        if task != "pusht":
            raise ValueError("A representation override is supported only for Push-T")
        configuration["task"]["representation"] = representation
        # An explicit run override takes precedence over YAML method overrides.
        for method in method_configs:
            method["representation"] = representation
    if task == "pusht":
        # Validate all configured choices before creating an output directory.
        for method in method_configs:
            module.effective_coordinates(configuration, method)
    available = {canonical_method(method["id"]): method for method in method_configs}
    selected = list(available) if methods is None else [canonical_method(name) for name in methods]
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(available):
        raise ValueError(f"Choose distinct methods from {list(available)}")
    starts = experiment_starts(task, module, configuration)
    if steps is not None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        starts = [(seed, start, steps) for seed, start, _ in starts]
        if task == "cartpole":
            configuration["steps"] = steps
        elif task == "pusht":
            configuration["task"]["max_episode_steps"] = steps
        else:
            configuration["evaluation"]["steps"] = steps
    expert_comparable = task != "pusht" and expert_comparison_matches(task, configuration)
    all_seeds = [start[0] for start in starts]
    seeds = all_seeds if seeds is None else list(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds).issubset(all_seeds):
        raise ValueError(f"Choose distinct seeds from {all_seeds}")
    if task == "cartpole":
        banks = configuration["datasets"]
        if len(banks) != 1:
            raise ValueError("Cartpole requires one shared demonstration bank")
        bank = next(iter(banks.values()))
    else:
        collection = configuration["demonstrations"]
        bank = collection.get("data_file") or collection["dataset_file"]
    bank_path, bank_hash = checked_input(bank)
    inputs = {str(bank_path.relative_to(ROOT)): bank_hash}
    starts_path, starts_hash = checked_input(starts_file(task, configuration))
    inputs[str(starts_path.relative_to(ROOT))] = starts_hash
    experts = {}
    if task != "pusht":
        experts, expert_hashes = saved_experts(task, configuration, starts)
        inputs.update(expert_hashes)
    else:
        manifest_path, manifest_hash = checked_input(configuration["demonstrations"]["manifest_file"])
        inputs[str(manifest_path.relative_to(ROOT))] = manifest_hash
    protocol = protocol_manifest(task, configuration, method_configs, starts, inputs)
    output = resolve(output or "runs/benchmark")
    initialize_output(output, task, protocol)
    data = load_dataset(bank_path, task)
    if task != "pusht":
        data = prepare_dataset(data)
    if task == "cartpole":
        cache = module.prepare_context(data, configuration["past"], configuration["future"],
                                       configuration["neighbors"], configuration["past_inputs"])
    else:
        cache = {False: {}, True: {}} if task == "reacher" else {}
    for identifier in selected:
        method = available[identifier]
        parameters = None
        if task != "pusht":
            if "grid" in method or "candidates" in method:
                raise ValueError(f"Fixed method {identifier} cannot contain a tuning grid")
            parameters = deepcopy(method["parameters"])
        for start in starts:
            seed = start[0]
            if seed not in seeds:
                continue
            destination = output / task / identifier / f"seed_{seed}"
            with file_lock(destination.parent / f".seed_{seed}.lock"):
                if completed_record(destination, protocol["sha256"]):
                    print(f"{task}/{identifier}/{seed}: verified existing record", flush=True)
                    continue
                metadata = {"protocol_sha256": protocol["sha256"], "expert_comparable": expert_comparable,
                            "config": configuration, "methods": [method], "method": method,
                            "test_starts": starts, "task": task, "seed": seed,
                            "timing_protocol": "Online telemetry; use scripts/timing.py for controlled timing"}
                with tempfile.TemporaryDirectory(prefix=f".seed_{seed}_", dir=destination.parent) as temporary:
                    staging = Path(temporary)
                    if task == "pusht":
                        row, arrays, decisions, queries, geometry = module.run_fixed_trial(
                            data, configuration, method, seed, cache)
                        row.update(task=task, method=identifier, evaluation_steps=start[2])
                        metadata.update(geometry=geometry, environment_seed=seed, policy_seed=row["policy_seed"],
                                        representation=row["representation"], normalization=row.pop("normalization", None),
                                        coordinate_convention={
                                            "observations": "Physical world coordinates",
                                            "policy_observations": "Configured controller coordinates before normalization",
                                            "actions": "Clipped world target positions sent to the simulator",
                                            "planning_anchor": "Last history block pose; decoded chunk fixed throughout execution",
                                        })
                        _atomic_write(staging / "trajectory.npz", lambda stream: np.savez_compressed(stream, **arrays))
                        atomic_json_lines(staging / "diagnostics.jsonl", decisions)
                        atomic_json_lines(staging / "queries.jsonl", queries)
                        _write_json(staging / "result.json", row)
                    else:
                        row, recorder = run_controller_trial(task, module, configuration, method, parameters, data, start, cache)
                        row.update(task=task, method=identifier, evaluation_steps=start[2])
                        metadata.update(paired_expert=experts[seed], reconstruction=recorder.reconstruction)
                        export_episode(staging, task, row, metadata, expert_row=experts[seed],
                                       references=recorder.references())
                        recorder.finalize(staging)
                        metadata = json.loads((staging / "metadata.json").read_text())
                        metadata["recording"] = {"render_state": "render_state.npz", "rendering": "rendering.json",
                                                 "frames": len(recorder.frames), "decisions": len(recorder.decisions),
                                                 "render_state_sha256": digest(staging / "render_state.npz"),
                                                 "assets": {path.name: digest(path) for path in (staging / "rendering_assets").glob("*")}}
                    publish_record(staging, destination, metadata)
                print(f"{task}/{identifier}/{seed}: {row.get('terminal_reason', row.get('termination_reason'))}", flush=True)
    from .reporting import aggregate_results
    with file_lock(output / ".report.lock"):
        return aggregate_results(output)


def main(task):
    parser = argparse.ArgumentParser(description=f"Run the configured {task} benchmark")
    parser.add_argument("command", nargs="?", choices=("run", "describe", "aggregate"), default="run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods", nargs="+", help="Method names, e.g. bd_chi2 select_dpc")
    parser.add_argument("--seeds", nargs="+", type=int, help="A subset of the configured test starts")
    parser.add_argument("--steps", type=int, help="Override the evaluation horizon")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    if task == "pusht":
        from ..tasks.pusht_coordinates import REPRESENTATIONS
        parser.add_argument("--representation", choices=REPRESENTATIONS,
                            help="Override the task and method representations for this run")
    arguments = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if arguments.command == "describe":
            configuration = load_experiment_config(task, arguments.config)
            module = import_module(f"experiments.{task}")
            description = {"task": task, "configuration": configuration,
                           "starts": experiment_starts(task, module, configuration)}
            if task == "pusht":
                if arguments.representation is not None:
                    configuration["task"]["representation"] = arguments.representation
                configured_methods = load_method_configs(task, configuration["methods"])
                description["representations"] = {
                    method["id"]: (arguments.representation or module.effective_coordinates(configuration, method).name)
                    for method in configured_methods}
            print(json.dumps(json_value(description), indent=2))
        elif arguments.command == "aggregate":
            from .reporting import aggregate_results
            aggregate_results(resolve(arguments.output))
        else:
            run_task(task, config_path=arguments.config, methods=arguments.methods, seeds=arguments.seeds,
                     steps=arguments.steps, output=arguments.output,
                     representation=getattr(arguments, "representation", None))
