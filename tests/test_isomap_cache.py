"""Independent persistent-Isomap cache checks; all artifacts use pytest tmp_path.

Run with the project virtual environment and single-threaded BLAS settings.
ISOMAP_REPO_ROOT can point to another checkout containing the implementation.
"""

from pathlib import Path
import importlib.util
import multiprocessing
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pytest


PROJECT_ROOT = Path(os.environ.get(
    "ISOMAP_REPO_ROOT", str(Path(__file__).resolve().parents[1]),
))
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.manifold import Isomap
from src.data import (
    DeePCConstraints, DeePCControllerArgs, DeePCCost, DeePCDims,
    TrajectoryData, TrajectoryDataSet,
)
from src.optimization.hankel import HankelMatrixGenerator
from src.policies.select_dpc import SelectDPC
from src.selectors.isomap import IsoMapEmbeddedDistances
from src.selectors import isomap_cache


def small_problem(count=64):
    """Persistently excited two-input LTI trajectories, with genuine Hankel columns."""
    inputs = np.random.default_rng(17).normal(size=(count, 2))
    outputs, state = [], 0.0
    for action in inputs:
        state = 0.8 * state + action[0] - 0.5 * action[1]
        outputs.append([state])
    data = TrajectoryDataSet(
        [TrajectoryData(outputs, inputs)], "cache-test", "post_action",
    )
    dimensions = DeePCDims(T_past=2, T_fut=2, p=1, m=2)
    matrices = HankelMatrixGenerator(2, 2).generate_hankel_matrices(data)
    return data, dimensions, matrices


def make_selector(problem, cache_dir=None, **options):
    data, dimensions, matrices = problem
    parameters = dict(n_neighbors=6, n_components=3, random_state=1)
    parameters.update(options)
    return IsoMapEmbeddedDistances(
        data, dimensions, hankel_matrices=matrices, cache_dir=cache_dir,
        **parameters,
    )


def query(selector, matrices):
    inputs, outputs = matrices
    # A non-training query avoids relying solely on cached training coordinates.
    return selector(
        0.6 * inputs[:, 1:2] + 0.4 * inputs[:, 4:5],
        0.6 * outputs[:, 1:2] + 0.4 * outputs[:, 4:5],
        inputs, outputs, None,
    )


def assert_same_selection(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])


def assert_same_rng(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.fixture
def fit_calls(monkeypatch):
    calls = []
    original = Isomap.fit

    def counted_fit(instance, *args, **kwargs):
        calls.append(len(args[0]))
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Isomap, "fit", counted_fit)
    return calls


def test_miss_and_hit_preserve_rankings_distances_and_controller_actions(tmp_path, fit_calls):
    problem = small_problem()
    uncached = make_selector(problem)
    missing = make_selector(problem, tmp_path)
    cached = make_selector(problem, tmp_path)
    assert len(fit_calls) == 2, "The third construction must load the fitted embedding"
    expected = query(uncached, problem[2])
    assert_same_selection(expected, query(missing, problem[2]))
    assert_same_selection(expected, query(cached, problem[2]))

    data, dimensions, matrices = problem
    costs = DeePCCost(np.eye(1), 0.1 * np.eye(2), 20.0, 0.01, 0.0)
    args = DeePCControllerArgs(
        data, dimensions, 64, costs, DeePCConstraints(None, None, None, None),
        np.zeros(2), False,
    )
    all_actions, all_selected = [], []
    for selector in (uncached, missing, cached):
        policy = SelectDPC(
            args, selector=selector, num_hankel_cols=24, n_iter=1,
            hankel_matrices=matrices,
            solver_options={"eps_abs": 1e-8, "eps_rel": 1e-8,
                            "max_iter": 20000, "adaptive_rho_interval": 25},
        )
        state = 0.0
        actions, selected = [], []
        for _ in range(3):
            action = policy.act(np.array([state]), np.array([1.0]))
            assert policy.diagnostics()["action_source"] == "optimized"
            actions.append(action.copy())
            selected.append(policy.diagnostics()["selected_indices"].copy())
            policy.record_applied_action(action)
            state = 0.8 * state + action[0] - 0.5 * action[1]
        all_actions.append(np.array(actions))
        all_selected.append(np.array(selected))
    for actions, selected in zip(all_actions[1:], all_selected[1:]):
        np.testing.assert_array_equal(actions, all_actions[0])
        np.testing.assert_array_equal(selected, all_selected[0])


@pytest.mark.parametrize("change", [
    "values", "column_order", "dtype", "dimensions", "neighbors", "components",
    "scaler", "subsample", "resampling_seed", "radius",
])
def test_cache_identity_changes_with_dictionary_or_fit_settings(tmp_path, fit_calls, change):
    problem = small_problem()
    make_selector(problem, tmp_path)
    data, dimensions, matrices = problem
    matrices = tuple(matrix.copy() for matrix in matrices)
    options = {}
    if change == "values":
        matrices[0][0, 0] += 0.125
    elif change == "column_order":
        matrices = tuple(matrix[:, ::-1].copy() for matrix in matrices)
    elif change == "dtype":
        matrices = tuple(matrix.astype(np.float32) for matrix in matrices)
    elif change == "dimensions":
        dimensions = DeePCDims(T_past=1, T_fut=3, p=1, m=2)
    elif change == "neighbors":
        options["n_neighbors"] = 7
    elif change == "components":
        options["n_components"] = 4
    elif change == "scaler":
        options["use_scaler"] = True
    elif change == "subsample":
        options["subsample_factor"] = 0.9
    elif change == "resampling_seed":
        options["random_state"] = 2
    elif change == "radius":
        options.update(n_neighbors=None, radius=1000.0)
    changed_problem = data, dimensions, matrices
    changed = make_selector(changed_problem, tmp_path, **options)
    assert len(fit_calls) == 2, f"{change} must invalidate the old embedding"
    again = make_selector(changed_problem, tmp_path, **options)
    assert len(fit_calls) == 2
    assert_same_selection(query(changed, matrices), query(again, matrices))


def test_corrupt_payload_is_refitted_and_replaced(tmp_path, fit_calls):
    problem = small_problem()
    expected = query(make_selector(problem, tmp_path), problem[2])
    payloads = [path for path in tmp_path.rglob("*")
                if path.is_file() and path.suffix in (".pkl", ".pickle", ".joblib")]
    assert len(payloads) == 1, "Expected one persistent embedding payload"
    payloads[0].write_bytes(b"interrupted or damaged cache payload")
    recovered = make_selector(problem, tmp_path)
    assert len(fit_calls) == 2
    assert_same_selection(expected, query(recovered, problem[2]))
    assert_same_selection(expected, query(make_selector(problem, tmp_path), problem[2]))
    assert len(fit_calls) == 2, "Recovery must publish a usable replacement"


def test_failed_fit_releases_lock_without_publishing_partial_state(tmp_path, monkeypatch):
    problem = small_problem()
    original = Isomap.fit
    calls = []

    def fail_once(instance, *args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Deliberate fit failure")
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Isomap, "fit", fail_once)
    with pytest.raises(RuntimeError, match="Deliberate fit failure"):
        make_selector(problem, tmp_path)
    recovered = make_selector(problem, tmp_path)
    cached = make_selector(problem, tmp_path)
    assert len(calls) == 2
    assert_same_selection(query(recovered, problem[2]), query(cached, problem[2]))


def _concurrent_client(cache_dir, start_event, output_queue):
    try:
        problem = small_problem()
        if not start_event.wait(10):
            raise TimeoutError("The concurrency test did not release its clients")
        selector = make_selector(problem, cache_dir)
        indices, distances = query(selector, problem[2])
        output_queue.put(("ok", indices, distances))
    except Exception:
        output_queue.put(("error", traceback.format_exc()))


def test_concurrent_processes_share_one_fit(tmp_path, monkeypatch):
    context = multiprocessing.get_context("fork")
    fit_count = context.Value("i", 0)
    original = Isomap.fit

    def slow_counted_fit(instance, *args, **kwargs):
        with fit_count.get_lock():
            fit_count.value += 1
        time.sleep(0.15)  # Make every other client arrive before the fit finishes.
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Isomap, "fit", slow_counted_fit)
    start_event, output_queue = context.Event(), context.Queue()
    clients = [context.Process(target=_concurrent_client,
                               args=(tmp_path, start_event, output_queue))
               for _ in range(4)]
    try:
        for client in clients:
            client.start()
        start_event.set()
        results = [output_queue.get(timeout=20) for _ in clients]
        for client in clients:
            client.join(timeout=10)
            assert client.exitcode == 0
        assert all(result[0] == "ok" for result in results), results
        assert fit_count.value == 1, "Concurrent misses must fit only once"
        for result in results[1:]:
            assert_same_selection(results[0][1:], result[1:])
    finally:
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join(timeout=5)
        output_queue.close()


def test_dense_cache_does_not_consume_or_restore_global_rng(tmp_path, fit_calls):
    problem = small_problem()
    np.random.seed(1234)
    before = np.random.get_state()
    uncached = make_selector(problem)
    assert_same_rng(before, np.random.get_state())
    missing = make_selector(problem, tmp_path)
    assert_same_rng(before, np.random.get_state())
    np.random.seed(9876)
    hit_before = np.random.get_state()
    cached = make_selector(problem, tmp_path)
    assert_same_rng(hit_before, np.random.get_state())
    assert len(fit_calls) == 2
    assert_same_selection(query(uncached, problem[2]), query(missing, problem[2]))
    assert_same_selection(query(uncached, problem[2]), query(cached, problem[2]))


def test_automatic_arpack_bypasses_cache_and_preserves_rng_semantics(tmp_path, fit_calls):
    problem = small_problem(224)  # 221 columns and 3 components trigger auto-ARPACK.
    np.random.seed(54321)
    before = np.random.get_state()
    uncached = make_selector(problem)
    after = np.random.get_state()
    assert not np.array_equal(before[1], after[1]) or before[2] != after[2]
    expected = query(uncached, problem[2])
    for _ in range(2):
        np.random.seed(54321)
        selector = make_selector(problem, tmp_path)
        assert_same_rng(after, np.random.get_state())
        assert_same_selection(expected, query(selector, problem[2]))
    assert len(fit_calls) == 3, "The RNG-dependent auto-ARPACK path must not be reused"


def test_ten_components_with_more_than_two_hundred_columns_remains_cacheable(tmp_path, fit_calls):
    problem = small_problem(224)
    np.random.seed(321)
    before = np.random.get_state()
    missing = make_selector(problem, tmp_path, n_components=10)
    cached = make_selector(problem, tmp_path, n_components=10)
    assert len(fit_calls) == 1, "The sklearn auto solver switches to dense at 10 components"
    assert_same_rng(before, np.random.get_state())
    assert_same_selection(query(missing, problem[2]), query(cached, problem[2]))


def test_unexpected_fit_rng_consumption_does_not_publish_a_cache(tmp_path, monkeypatch):
    problem = small_problem()
    original = Isomap.fit
    calls = []

    def rng_consuming_fit(instance, *args, **kwargs):
        calls.append(True)
        np.random.random()
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Isomap, "fit", rng_consuming_fit)
    np.random.seed(111)
    np.random.random(2)
    expected_rng = np.random.get_state()
    np.random.seed(111)
    make_selector(problem, tmp_path)
    make_selector(problem, tmp_path)
    assert len(calls) == 2, "An unexpectedly RNG-dependent fit must never be reused"
    assert_same_rng(expected_rng, np.random.get_state())
    assert not list(tmp_path.rglob("*.joblib"))


@pytest.mark.parametrize("runtime_change", ["dependency_version", "thread_count"])
def test_runtime_fingerprint_prevents_reusing_an_incompatible_fit(
    tmp_path, monkeypatch, fit_calls, runtime_change,
):
    problem = small_problem()
    make_selector(problem, tmp_path)
    if runtime_change == "dependency_version":
        monkeypatch.setattr(isomap_cache.joblib, "__version__", "99.0.0")
    else:
        pools = [dict(pool) for pool in isomap_cache.threadpool_info()]
        assert pools, "The test environment should expose its BLAS thread pool"
        pools[0]["num_threads"] += 1
        monkeypatch.setattr(isomap_cache, "threadpool_info", lambda: pools)
    changed = make_selector(problem, tmp_path)
    assert len(fit_calls) == 2
    cached = make_selector(problem, tmp_path)
    assert len(fit_calls) == 2
    assert_same_selection(query(changed, problem[2]), query(cached, problem[2]))


def test_source_identity_describes_imported_code_not_later_on_disk_edits(tmp_path):
    """An old live process must not publish its fits under a newer source identity."""
    module_directory = tmp_path / "source"
    module_directory.mkdir()
    for name in ("isomap.py", "isomap_cache.py"):
        shutil.copyfile(PROJECT_ROOT / "src" / "selectors" / name, module_directory / name)

    def load_helper(name):
        spec = importlib.util.spec_from_file_location(name, module_directory / "isomap_cache.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    imported = load_helper("original_cache_helper")
    _, dimensions, matrices = small_problem()
    indices = np.arange(matrices[0].shape[1])
    options = {"n_components": 3}
    old_key, _ = imported._identity(matrices, indices, dimensions, options)
    source = module_directory / "isomap.py"
    source.write_text(source.read_text() + "\n# Simulate an edit after the old process imported its code.\n")
    still_old_key, _ = imported._identity(matrices, indices, dimensions, options)
    assert still_old_key == old_key
    freshly_imported = load_helper("updated_cache_helper")
    new_key, _ = freshly_imported._identity(matrices, indices, dimensions, options)
    assert new_key != old_key


def test_none_seed_cache_hit_consumes_the_normal_global_resampling_rng(tmp_path, fit_calls):
    problem = small_problem()
    np.random.seed(2026)
    uncached = make_selector(problem, random_state=None)
    expected_rng = np.random.get_state()
    expected_selection = query(uncached, problem[2])
    for expected_status in ("miss", "hit"):
        np.random.seed(2026)
        selector = make_selector(problem, tmp_path, random_state=None)
        assert selector.cache_info["status"] == expected_status
        assert_same_rng(expected_rng, np.random.get_state())
        assert_same_selection(expected_selection, query(selector, problem[2]))
    assert len(fit_calls) == 2, "The hit must still resample, but must not refit"


def test_mutable_randomstate_cache_hit_advances_normally_and_new_permutation_misses(
    tmp_path, fit_calls,
):
    problem = small_problem()
    np.random.seed(345)
    global_before = np.random.get_state()
    reference_rng = np.random.RandomState(2026)
    uncached = make_selector(problem, random_state=reference_rng)
    first_rng_state = reference_rng.get_state()
    first_selection = query(uncached, problem[2])

    missing_rng = np.random.RandomState(2026)
    missing = make_selector(problem, tmp_path, random_state=missing_rng)
    assert missing.cache_info["status"] == "miss"
    assert_same_rng(first_rng_state, missing_rng.get_state())
    assert_same_selection(first_selection, query(missing, problem[2]))

    cached_rng = np.random.RandomState(2026)
    cached = make_selector(problem, tmp_path, random_state=cached_rng)
    assert cached.cache_info["status"] == "hit"
    assert_same_rng(first_rng_state, cached_rng.get_state())
    assert_same_selection(first_selection, query(cached, problem[2]))
    assert len(fit_calls) == 2

    # Both generators have now drawn once. Their next draw is a different,
    # matching permutation, which must get its own embedding and cache entry.
    next_uncached = make_selector(problem, random_state=reference_rng)
    next_missing = make_selector(problem, tmp_path, random_state=cached_rng)
    assert next_missing.cache_info["status"] == "miss"
    assert next_missing.cache_info["key"] != missing.cache_info["key"]
    assert_same_rng(reference_rng.get_state(), cached_rng.get_state())
    assert_same_selection(query(next_uncached, problem[2]), query(next_missing, problem[2]))
    assert len(fit_calls) == 4
    assert_same_rng(global_before, np.random.get_state())


def test_library_discovery_order_does_not_prevent_cache_reuse(tmp_path, monkeypatch, fit_calls):
    problem = small_problem()
    pools = isomap_cache.threadpool_info()
    assert len(pools) > 1
    missing = make_selector(problem, tmp_path)
    monkeypatch.setattr(isomap_cache, "threadpool_info", lambda: list(reversed(pools)))
    cached = make_selector(problem, tmp_path)
    assert cached.cache_info["status"] == "hit"
    assert missing.cache_info["key"] == cached.cache_info["key"]
    assert len(fit_calls) == 1
    assert_same_selection(query(missing, problem[2]), query(cached, problem[2]))
