"""Replay saved controller queries and record process-CPU time per action.

Setup, warm-up, logging, validation, and simulator initialization are excluded.
Saved actions are checked against the current implementation. Partial replays
are labelled and excluded from the complete-sequence timing summary.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

METHODS = ("bd_chi2", "select_dpc", "bd_mean", "bd_sample", "bd_kl", "bd_mean_chi2", "windowed_deepc", "deepc")
TASKS = ("cartpole", "reacher", "rocket")
ACTION_TOLERANCE = 1e-5


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_nonfinite_values(encoded):
    """Decode only explicit JSON-pointer tags; a plain null is never infinity."""
    restored = deepcopy(encoded)
    tags = encoded.get('nonfinite_values', {})
    if not isinstance(tags, dict):
        raise ValueError('nonfinite_values must be a JSON-pointer mapping')
    replacements = {'+Infinity': float('inf'), '-Infinity': float('-inf'), 'NaN': float('nan')}

    def key(container, token, pointer):
        if isinstance(container, dict):
            if token not in container:
                raise ValueError(f'Nonfinite pointer does not exist: {pointer}')
            return token
        if isinstance(container, list):
            if not token.isascii() or not token.isdigit() or (len(token) > 1 and token.startswith('0')):
                raise ValueError(f'Invalid nonfinite array index: {pointer}')
            index = int(token)
            if index >= len(container):
                raise ValueError(f'Nonfinite array index is out of range: {pointer}')
            return index
        raise ValueError(f'Nonfinite pointer traverses a scalar: {pointer}')

    for pointer, marker in tags.items():
        if not isinstance(pointer, str) or not pointer.startswith('/'):
            raise ValueError(f'Invalid nonfinite JSON pointer: {pointer!r}')
        if not isinstance(marker, str) or marker not in replacements:
            raise ValueError(f'Unknown nonfinite marker at {pointer}: {marker!r}')
        tokens = []
        for token in pointer[1:].split('/'):
            if any(not suffix or suffix[0] not in '01' for suffix in token.split('~')[1:]):
                raise ValueError(f'Invalid JSON-pointer escape: {pointer}')
            tokens.append(token.replace('~1', '/').replace('~0', '~'))
        container = restored
        for token in tokens[:-1]:
            container = container[key(container, token, pointer)]
        last = key(container, tokens[-1], pointer)
        if container[last] is not None:
            raise ValueError(f'Nonfinite tag must point to encoded null: {pointer}')
        container[last] = replacements[marker]
    return restored


def selected_parameters(metadata, result):
    if result.get('parameters'):
        return deepcopy(result['parameters']), 'saved_episode_parameters'
    if metadata.get('parameters'):
        return deepcopy(metadata['parameters']), 'saved_metadata_parameters'
    method = metadata['methods'][0]
    if metadata['config'].get('evaluation_mode') != 'fixed' or method.get('grid'):
        raise ValueError('Missing selected parameters outside an unambiguous fixed profile')
    candidates = method.get('candidates', [{}])
    if len(candidates) != 1:
        raise ValueError('Missing selected parameters with more than one candidate')
    return {**deepcopy(method['parameters']), **deepcopy(candidates[0])}, 'unique_fixed_method_profile'


def encode_parameters(parameters):
    tags = {}
    def encode(value, pointer):
        if isinstance(value, float) and not np.isfinite(value):
            tags[pointer] = 'NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
            return None
        if isinstance(value, dict):
            return {key: encode(item, pointer + '/' + str(key).replace('~', '~0').replace('/', '~1'))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item, pointer + '/' + str(index)) for index, item in enumerate(value)]
        return value
    return encode(parameters, '/parameters'), tags


def initialize_policy(module, task_name, method, parameters, metadata, arrays, seed, data_path):
    """Construct and reset the task before replaying the saved controller queries."""
    from src.dataset_io import load_dataset
    from src.data import prepare_dataset
    config = deepcopy(metadata['config'])
    task = None
    try:
        if task_name == 'cartpole':
            data = load_dataset(data_path, 'cartpole')
            context = module.prepare_context(data, config['past'], config['future'],
                                             config['neighbors'], config['past_inputs'])
            policy = module.build_policy(context, method, parameters)
            task = module.CartpoleTask(**config['task'])
            initial = task.reset(seed)
            np.testing.assert_array_equal(initial, arrays['states'][0])
            bounds = np.array([-3.]), np.array([3.])
            augmented = False
            order = 'build policy; construct task; task.reset(seed); policy.reset(seed)'
        elif task_name == 'reacher':
            config['demonstrations']['data_file'] = str(data_path)
            data = prepare_dataset(load_dataset(data_path, "reacher"))
            deepc = method['family'] in module.DEEPC_FAMILIES
            policy_data = module.strip_targets(data) if deepc else data
            # evaluate() constructs the policy BEFORE constructing/resetting the
            # episode task. Its norm selector does not use NumPy's global RNG.
            policy = module.build_policy(method, parameters, policy_data, config, cache={})
            starts = [item for item in metadata['test_starts'] if int(item[0]) == seed]
            if len(starts) != 1:
                raise ValueError('Reacher metadata does not identify exactly one matching start')
            _, targets, original_cap = starts[0]
            task = module.ReacherTask(**{**config.get('task', {}), 'targets': targets,
                                        'max_episode_steps': int(original_cap)})
            initial = task.reset(seed)
            np.testing.assert_array_equal(task.observation(initial), arrays['observations'][0])
            np.testing.assert_array_equal(initial, arrays['raw_observations'][0])
            np.testing.assert_array_equal(task.targets, arrays['targets'])
            limit = float(config['action_limit'])
            bounds = np.full(2, -limit), np.full(2, limit)
            augmented = not deepc
            order = 'build policy; construct episode task; task.reset(seed); policy.reset(seed); no extra global RNG seeding'
        else:
            config['demonstrations']['dataset_file'] = str(data_path)
            data = prepare_dataset(load_dataset(data_path, 'rocket'), config['demonstrations'].get('episode_order', 'upstream_filename'))
            starts = [item for item in metadata['test_starts'] if int(item[0]) == seed]
            if len(starts) != 1:
                raise ValueError('Rocket metadata does not identify exactly one matching start')
            task = module.make_task(starts[0][1], config['evaluation']['steps'], config)
            np.random.seed(seed)
            initial = task.reset(seed)
            np.testing.assert_array_equal(task.observation(initial), arrays['observations'][0])
            np.testing.assert_array_equal(initial, arrays['raw_observations'][0])
            policy = module.build_policy(method, parameters, data, task, cache={})
            bounds = module.bounds(task)
            augmented = False
            order = 'construct task; np.random.seed(seed); task.reset(seed); build policy; policy.reset(seed)'
        policy.reset(seed)
        return policy, task, bounds, augmented, order
    except BaseException:
        if task is not None:
            task.close()
        raise


def warm_cpu(seconds):
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    value = 1
    while time.process_time() - cpu_started < seconds:
        for _ in range(1024):
            value = (value * 1664525 + 1013904223) & 0xffffffff
    return dict(cpu_seconds=time.process_time() - cpu_started,
                wall_seconds=time.perf_counter() - wall_started, checksum=value,
                kind='deterministic integer arithmetic; no policy or RNG calls')


def replay_sequence(policy, arrays, bounds, augmented, feedback_function, *, max_decisions=None, progress=None):
    """Sequential stateful replay; simulator-independent and directly testable."""
    count = len(arrays['actions'])
    mask = np.asarray(arrays.get('policy_decision_mask', np.ones(count, dtype=bool)), dtype=bool)
    feedback = np.asarray(arrays.get('feedback_actions', arrays.get('applied_actions', arrays['actions'])))
    if mask.shape != (count,) or feedback.shape != arrays['actions'].shape:
        raise ValueError('Decision mask or feedback shape differs from the saved action history')
    if arrays['observations'].shape[0] != count + 1 or arrays['references'].shape[0] != count:
        raise ValueError('Observations/references are not aligned with the saved actions')
    expected = arrays.get('proposed_actions')
    if expected is not None and expected.shape != arrays['actions'].shape:
        raise ValueError('Recorded proposal shape differs from the action history')
    indices, cpu_ns, wall_ns, gaps, exact, replayed, expected_values = [], [], [], [], [], [], []
    processed_steps = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for index in range(count):
            if mask[index] and max_decisions is not None and len(indices) >= max_decisions:
                break
            if mask[index]:
                observation = arrays['observations'][index].copy()
                reference = arrays['references'][index].copy()
                if augmented:
                    observation = np.r_[observation, reference[4:6]]
                if progress:
                    progress(dict(stage='policy_act', step=index, completed_decisions=len(indices),
                                  active_call_started_utc=utc(), active_call_started_monotonic=time.monotonic()))
                # Feature construction, logging and validation are outside these
                # clocks. These enclose precisely the controller's action call.
                wall_started = time.perf_counter_ns()
                cpu_started = time.process_time_ns()
                proposal = policy.act(observation, reference)
                cpu_elapsed = time.process_time_ns() - cpu_started
                wall_elapsed = time.perf_counter_ns() - wall_started
                action = np.asarray(proposal, dtype=float).reshape(-1)
                if action.shape != arrays['actions'][index].shape or not np.isfinite(action).all():
                    raise RuntimeError(f'Invalid replay action at step {index}')
                comparison = action if expected is not None else np.clip(action, *bounds)
                target = expected[index] if expected is not None else feedback[index]
                if not np.isfinite(target).all():
                    raise ValueError(f'Nonfinite saved comparison action at step {index}')
                gap = float(np.max(np.abs(comparison - target)))
                indices.append(index); cpu_ns.append(cpu_elapsed); wall_ns.append(wall_elapsed)
                gaps.append(gap); exact.append(bool(np.array_equal(comparison, target)))
                replayed.append(action.copy()); expected_values.append(target.copy())
            # Apply feedback on every step, including task overrides.
            feedback_function(policy, feedback[index])
            processed_steps = index + 1
    dimension = arrays['actions'].shape[1]
    return dict(step_indices=np.asarray(indices, dtype=np.int64), cpu_ns=np.asarray(cpu_ns, dtype=np.int64),
                wall_ns=np.asarray(wall_ns, dtype=np.int64), absolute_action_gaps=np.asarray(gaps),
                exact_action_matches=np.asarray(exact, dtype=bool),
                replay_actions=np.asarray(replayed).reshape(-1, dimension),
                expected_actions=np.asarray(expected_values).reshape(-1, dimension)), processed_steps


def canonical_data(task, metadata, config, records):
    """Verify the demonstration bank against the recorded run protocol."""
    path = ROOT / 'data' / task / 'demonstrations.npz'
    actual = sha256(path)
    protocol = json.loads((records / "protocols" / f"{task}.json").read_text())
    declared_hash = protocol.pop("sha256")
    actual_protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if actual_protocol_hash != declared_hash or declared_hash != metadata["protocol_sha256"]:
        raise ValueError("Record protocol hash mismatch")
    expected = protocol["input_sha256"][str(path.relative_to(ROOT))]
    if actual != expected:
        raise ValueError(f'Canonical {task} data differs from its recorded bank')
    return path, actual


def cpu_siblings(cpu):
    """Read Linux topology instead of assuming a fixed logical-CPU numbering."""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    siblings = set()
    for segment in path.read_text().strip().split(","):
        limits = [int(value) for value in segment.split("-")]
        siblings.update(range(limits[0], limits[-1] + 1))
    return sorted(siblings)


def resource_snapshot(cpus):
    counters = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {f"cpu{cpu}" for cpu in cpus}:
            counters[fields[0]] = [int(value) for value in fields[1:]]
    frequencies = {}
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        frequencies[str(cpu)] = int(path.read_text()) if path.exists() else None
    return {"utc": utc(), "monotonic": time.monotonic(), "cpu_jiffies": counters,
            "frequency_khz": frequencies, "load_average": list(os.getloadavg())}


def monitor_resources(output, cpus):
    """Monitor in a separate process, keeping its CPU time outside action timings."""
    running = True

    def stop(_signal, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    with Path(output).open("w") as stream:
        while running:
            stream.write(json.dumps(resource_snapshot(cpus)) + "\n")
            stream.flush()
            time.sleep(0.5)


def replay(task_name, method_name, seed, records, output, *, cpu,
           max_decisions=None, warmup_seconds=2.0):
    record = records / task_name / method_name / f"seed_{seed}"
    output = output / task_name / method_name / f"seed_{seed}"
    if (output / "result.json").exists():
        raise ValueError(f"Use a fresh output directory; timing result already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    original_affinity = sorted(os.sched_getaffinity(0))
    if cpu not in original_affinity:
        raise ValueError(f"CPU {cpu} is outside the permitted affinity: {original_affinity}")
    siblings = cpu_siblings(cpu)
    monitor = None
    task = None
    report = {"task": task_name, "method": method_name, "seed": seed,
              "started_utc": utc(), "source_record": str(record.resolve()),
              "pilot": max_decisions is not None, "max_decisions": max_decisions,
              "cpu": cpu, "smt_siblings": siblings,
              "validated_replay": False, "initialization_included": False,
              "clocks": {"primary": "process_time_ns", "secondary": "perf_counter_ns"},
              "load_qualification": "Affinity-controlled with recorded load; CPU reservation is not enforced.",
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                for path in sorted((ROOT / "src").rglob("*.py"))},
              "timing_script_sha256": sha256(Path(__file__))}
    report["platform"] = platform.platform()
    report["python"] = platform.python_version()
    report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in
        Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    report["packages"] = {}
    for package in ("numpy", "scipy", "cvxpy", "osqp", "scikit-learn", "gymnasium", "mujoco"):
        try:
            report["packages"][package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            pass
    try:
        metadata = restore_nonfinite_values(json.loads((record / "metadata.json").read_text()))
        original_result = restore_nonfinite_values(json.loads((record / "result.json").read_text()))
        if sha256(record / "trajectory.npz") != metadata["trajectory_sha256"]:
            raise ValueError("Saved trajectory differs from its metadata")
        with np.load(record / "trajectory.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        count = len(arrays["actions"])
        decisions = int(np.asarray(arrays.get("policy_decision_mask", np.ones(count, dtype=bool))).sum())
        parameters, origin = selected_parameters(metadata, original_result)
        encoded_parameters, parameter_tags = encode_parameters(parameters)
        data_path, data_hash = canonical_data(task_name, metadata, metadata["config"], records)
        report.update(parameters=encoded_parameters, parameters_source=origin,
                      nonfinite_values=parameter_tags, recorded_steps=count,
                      recorded_policy_decisions=decisions, data_sha256=data_hash,
                      input_sha256={name: sha256(record / name) for name in
                                    ("trajectory.npz", "metadata.json", "result.json")})
        if not decisions:
            report.update(state="no_actions", decisions=0, cpu_ms_per_action=None,
                          wall_ms_per_action=None, complete_sequence=True,
                          reason="No applied policy decisions; failed attempts are not action timings.")
            save_json(output / "result.json", report)
            return report
        monitor_command = [sys.executable, str(Path(__file__).resolve()), "--monitor",
                           str(output / "resources.jsonl"), "--monitor-cpus", *map(str, siblings)]
        monitor = subprocess.Popen(monitor_command, stdout=subprocess.DEVNULL)
        monitor_cpus = set(original_affinity) - set(siblings)
        if monitor_cpus:
            os.sched_setaffinity(monitor.pid, monitor_cpus)
        os.sched_setaffinity(0, {cpu})
        report["affinity"] = sorted(os.sched_getaffinity(0))
        with threadpool_limits(limits=1):
            module = importlib.import_module(f"experiments.{task_name}")
            start = time.perf_counter()
            policy, task, bounds, augmented, order = initialize_policy(
                module, task_name, metadata["methods"][0], parameters, metadata, arrays, seed, data_path)
            report.update(initialization_seconds=time.perf_counter() - start,
                          initialization_order=order, threadpools=threadpool_info())
            if any(pool["num_threads"] != 1 for pool in report["threadpools"]):
                raise RuntimeError("Numerical libraries must use one thread")
            report["warmup"] = warm_cpu(warmup_seconds)
            from src.evaluation import record_applied_action

            def progress(values):
                save_json(output / "progress.json", {"task": task_name, "method": method_name,
                          "seed": seed, "updated_utc": utc(), **values})

            measured, processed = replay_sequence(policy, arrays, bounds, augmented,
                record_applied_action, max_decisions=max_decisions, progress=progress)
            task.close()
            task = None
        np.savez_compressed(output / "decisions.npz", **measured)
        accepted = bool(np.all(measured["absolute_action_gaps"] <= ACTION_TOLERANCE))
        complete = processed == count and len(measured["cpu_ns"]) == decisions
        report.update(state="completed" if accepted else "action_mismatch",
                      complete_sequence=complete, all_actions_validated=accepted,
                      validated_replay=accepted,
                      decisions=len(measured["cpu_ns"]), processed_steps=processed,
                      cpu_ms_per_action=float(np.mean(measured["cpu_ns"])) / 1e6,
                      wall_ms_per_action=float(np.mean(measured["wall_ns"])) / 1e6,
                      exact_action_matches=int(measured["exact_action_matches"].sum()),
                      max_action_absolute_gap=float(measured["absolute_action_gaps"].max()),
                      action_absolute_tolerance=ACTION_TOLERANCE,
                      decisions_sha256=sha256(output / "decisions.npz"), finished_utc=utc())
        save_json(output / "result.json", report)
        if not accepted:
            raise ValueError(f"Controller action replay differs: {report['max_action_absolute_gap']}")
        return report
    except BaseException as error:
        report.update(state="failed", error=repr(error), validated_replay=False, finished_utc=utc())
        save_json(output / "result.json", report)
        raise
    finally:
        if task is not None:
            task.close()
        os.sched_setaffinity(0, original_affinity)
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/benchmark")
    parser.add_argument("--output", type=Path, help="Timing output directory (default: <root>/timing)")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--warmup-seconds", type=float, default=2.)
    parser.add_argument("--monitor", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--monitor-cpus", nargs="+", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.monitor:
        monitor_resources(args.monitor, args.monitor_cpus)
        return
    if not args.task or not args.method:
        parser.error("--task and --method are required")
    args.output = args.output or args.root / "timing"
    if args.max_decisions is not None and args.max_decisions <= 0:
        parser.error("--max-decisions must be positive")
    if not np.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("--warmup-seconds must be finite and nonnegative")
    available = sorted(int(path.parent.name.removeprefix("seed_"))
                       for path in (args.root / args.task / args.method).glob("seed_*/result.json"))
    seeds = args.seeds or available[:3]
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds).issubset(available):
        parser.error(f"Choose distinct saved seeds from {available}")
    results = []
    for seed in seeds:
        result = replay(args.task, args.method, seed, args.root.resolve(), args.output.resolve(),
                        cpu=args.cpu, max_decisions=args.max_decisions, warmup_seconds=args.warmup_seconds)
        results.append(result)
        print(json.dumps({key: result.get(key) for key in
                          ("task", "method", "seed", "state", "decisions", "cpu_ms_per_action")}), flush=True)
    valid = [result["cpu_ms_per_action"] for result in results if result["validated_replay"] and result.get("complete_sequence") and not result["pilot"]]
    summary = {"task": args.task, "method": args.method, "seeds": list(seeds),
               "all_sequences_validated": len(valid) == len(seeds),
               "mean_ms": float(np.mean(valid)) if valid else None,
               "sample_sd_ms": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
               "n": len(valid), "aggregation": "Equally weighted per-start process-CPU means"}
    save_json(args.output / args.task / args.method / "summary.json", summary)


if __name__ == "__main__":
    main()
