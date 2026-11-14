"""Rank dictionary columns by L1 distance in an Isomap embedding."""

import numpy as np
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from ..optimization.hankel import HankelMatrixGenerator
from .isomap_cache import fit_embedding


class IsoMapEmbeddedDistances:
    """``random_state`` seeds dictionary resampling, not the entire embedding.

    As upstream, Isomap uses sklearn's automatic eigensolver. Its ARPACK path
    also consumes NumPy's global RNG, which the experiment must seed.
    """

    def __init__(
        self, data, deepc_dims, use_scaler=False, n_neighbors=4,
        n_components=75, subsample_factor=1.0, radius=None, random_state=1,
        hankel_matrices=None, cache_dir=None,
    ):
        self.deepc_dims = deepc_dims
        if not 0 < subsample_factor <= 1:
            raise ValueError("subsample_factor must be in (0, 1]")
        if hankel_matrices is None:
            hankel_matrices = HankelMatrixGenerator(
                deepc_dims.T_past, deepc_dims.T_fut,
            ).generate_hankel_matrices(data)
        self._H_u, self._H_y = (np.array(matrix, copy=True) for matrix in hankel_matrices)
        count = self._H_u.shape[1]
        # Keep the original sampled row order, but retain its dictionary indices.
        self._indices = resample(
            np.arange(count), replace=False,
            n_samples=int(count * subsample_factor), random_state=random_state,
        )
        dictionary = np.vstack((self._H_u, self._H_y), dtype=np.float32)
        self._embedding = make_pipeline(
            StandardScaler() if use_scaler else None,
            Isomap(n_neighbors=n_neighbors, radius=radius,
                   n_components=n_components, p=1, n_jobs=-1),
        )
        options = dict(use_scaler=use_scaler, n_neighbors=n_neighbors,
                       n_components=n_components, subsample_factor=subsample_factor,
                       radius=radius, random_state=random_state)
        self._embedding, self.cache_info = fit_embedding(
            self._embedding, dictionary[:, self._indices].T,
            (self._H_u, self._H_y), self._indices, deepc_dims, options, cache_dir,
        )
        self._n_neighbors, self._use_scaler = n_neighbors, use_scaler

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        if not (np.array_equal(H_u, self._H_u) and np.array_equal(H_y, self._H_y)):
            raise ValueError("Isomap must be fitted to the same ordered dictionary used by the policy")
        query = np.append(input_traj, state_traj).reshape(1, -1)
        embedded_query = self._embedding.transform(query)
        norms = np.linalg.norm(
            self._embedding.steps[-1][-1].embedding_ - embedded_query, ord=1, axis=1,
        )
        # Scores and returned indices refer to actual Hankel columns, not shuffled rows.
        distances = np.full(self._H_u.shape[1], np.inf)
        distances[self._indices] = norms
        return self._indices[np.argsort(norms)], distances

    def get_selector_name(self):
        return f"{'scaled_' if self._use_scaler else ''}isomap_embedder_nn{self._n_neighbors}"
