"""Persistent fitted Isomap pipelines for deterministic, dense eigensolves.

Only use cache directories owned by this application: joblib stores Python
objects. The checksum detects incomplete/corrupt local files, not hostile input.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _same_rng_state(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _source_hashes():
    try:
        return {name: _file_hash(Path(__file__).with_name(name))
                for name in ['isomap.py', 'isomap_cache.py']}
    except OSError:
        return None


# A process keeps executing its imported code after files change on disk. Pin
# that identity now, rather than labelling an old fit with a newer file's hash.
_IMPORTED_SOURCE_HASHES = _source_hashes()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.random.RandomState):
        # The already drawn, ordered indices are hashed separately.
        return {'type': 'RandomState'}
    raise TypeError(f'Unsupported cache setting: {type(value).__name__}')


def _identity(matrices, indices, dimensions, options):
    runtime = [{key: pool.get(key) for key in
                ['user_api', 'internal_api', 'prefix', 'version', 'num_threads', 'architecture']}
               for pool in threadpool_info()]
    # Library discovery order depends on imports, not the numerical runtime.
    runtime.sort(key=lambda pool: json.dumps(pool, sort_keys=True))
    description = {
        'schema': 1, 'dimensions': vars(dimensions), 'options': options,
        'versions': {'python': list(sys.version_info[:2]), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'threadpools': runtime,
        'source': _IMPORTED_SOURCE_HASHES,
    }
    digest = hashlib.sha256(json.dumps(description, sort_keys=True, default=_json_value).encode())
    for array in [*matrices, indices]:
        descriptor = {'shape': array.shape, 'dtype': array.dtype.str, 'strides': array.strides}
        digest.update(json.dumps(descriptor, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest(), description


def _load(directory, key):
    """An invalid local entry is a miss; it never becomes a partial estimator."""
    try:
        metadata = json.loads((directory / 'metadata.json').read_text())
        payload = directory / 'fit.joblib'
        if metadata['key'] != key or metadata['payload_sha256'] != _file_hash(payload):
            return None
        fitted = joblib.load(payload, mmap_mode='r')
        isomap = fitted.steps[-1][1]
        if not all(hasattr(isomap, name) for name in ['nbrs_', 'dist_matrix_', 'kernel_pca_', 'embedding_']):
            return None
        return fitted
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError):
        return None


def _publish(root, directory, key, description, fitted):
    with tempfile.TemporaryDirectory(prefix=key + '.tmp.', dir=root) as temporary:
        temporary = Path(temporary)
        payload = temporary / 'fit.joblib'
        joblib.dump(fitted, payload, compress=0)
        metadata = {'key': key, 'identity': description, 'payload_sha256': _file_hash(payload),
                    'created_at': datetime.now(timezone.utc).isoformat()}
        (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=_json_value) + '\n')
        # Readers and writers both hold the same key lock. An invalid old entry
        # can be removed before the complete replacement is renamed into place.
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)


def fit_embedding(pipeline, training, matrices, indices, dimensions, options, cache_dir):
    """Return a fitted pipeline and small cache diagnostics; never replay RNG state."""
    if cache_dir is None:
        pipeline.fit(training)
        return pipeline, {'status': 'disabled'}
    components = options['n_components']
    components = len(training) if components is None else min(components, len(training))
    # sklearn's automatic KernelPCA solver uses randomized ARPACK initialization
    # for this branch. Leave its fit and RNG consumption exactly as before.
    if len(training) > 200 and components < 10:
        pipeline.fit(training)
        return pipeline, {'status': 'bypass_stochastic_solver'}
    if _IMPORTED_SOURCE_HASHES is None:
        pipeline.fit(training)
        return pipeline, {'status': 'bypass_source_unavailable'}

    key, description = _identity(matrices, indices, dimensions, options)
    root = Path(cache_dir).expanduser().resolve()
    directory = root / key
    info = {'key': key, 'directory': str(directory)}
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / (key + '.lock')).open('a+b')
    except OSError as error:
        warnings.warn(f'Isomap cache unavailable; fitting normally: {error}', RuntimeWarning)
        pipeline.fit(training)
        return pipeline, {**info, 'status': 'unavailable'}

    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as error:
            warnings.warn(f'Isomap cache locking unavailable; fitting normally: {error}', RuntimeWarning)
            pipeline.fit(training)
            return pipeline, {**info, 'status': 'unavailable'}
        fitted = _load(directory, key)
        if fitted is not None:
            return fitted, {**info, 'status': 'hit'}
        before = np.random.get_state()
        pipeline.fit(training)
        if not _same_rng_state(before, np.random.get_state()):
            # A changed dependency may introduce randomness even outside the
            # known ARPACK branch. Keep the fit and its normal RNG side effects.
            return pipeline, {**info, 'status': 'bypass_rng_changed'}
        try:
            _publish(root, directory, key, description, pipeline)
        except OSError as error:
            warnings.warn(f'Isomap fit succeeded but cache could not be saved: {error}', RuntimeWarning)
            return pipeline, {**info, 'status': 'write_failed'}
        return pipeline, {**info, 'status': 'miss'}
