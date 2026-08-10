import anndata as ad
import numpy as np
import palantir
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans


def _dense_matrix(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("adata.X must be a 2D matrix")
    return matrix


def _window_starts(n_obs: int, window_size: int, step_size: int) -> list[int]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    effective_window = min(window_size, n_obs)
    starts = list(range(0, max(1, n_obs - effective_window + 1), step_size)) or [0]
    final_start = max(0, n_obs - effective_window)
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def run_palantir(
    adata: ad.AnnData,
    start_idx: int,
    terminal_states: list[int | str] | None = None,
    n_diff_components: int = 10,
    knn: int = 30,
    num_waypoints: int = 500,
    scale_components: bool = True,
    max_iterations: int = 25,
    seed: int = 42,
    use_early_cell_as_start: bool = True,
    store_prefix: str = "palantir",
):
    if adata.n_obs < 2 or adata.n_vars < 1:
        raise ValueError("Palantir requires at least two observations and one feature")
    if not (0 <= start_idx < adata.n_obs):
        raise IndexError(f"start_idx out of range: {start_idx}")

    data_df = pd.DataFrame(_dense_matrix(adata), index=adata.obs_names, columns=adata.var_names)
    internal_index = data_df.index
    early_cell = internal_index[start_idx]

    if terminal_states is not None:
        terminal_states_internal = []
        for x in terminal_states:
            if isinstance(x, int):
                if not (0 <= x < adata.n_obs):
                    raise IndexError(f"terminal state index out of range: {x}")
                terminal_states_internal.append(internal_index[x])
            else:
                matches = np.where(adata.obs_names == x)[0]
                if len(matches) == 0:
                    raise KeyError(f"terminal state name not found: {x}")
                terminal_states_internal.append(internal_index[matches[0]])
        terminal_states = terminal_states_internal

    dm_res = palantir.utils.run_diffusion_maps(
        data_df,
        n_components=n_diff_components,
        knn=knn,
        seed=seed,
    )

    n_eigs = min(n_diff_components + 1, dm_res["EigenVectors"].shape[1])
    if n_eigs < 2:
        raise ValueError("Too few diffusion eigenvectors were produced.")

    ms_data = palantir.utils.determine_multiscale_space(
        dm_res,
        n_eigs=n_eigs,
    )

    pr_res = palantir.core.run_palantir(
        ms_data,
        early_cell=early_cell,
        terminal_states=terminal_states,
        knn=knn,
        num_waypoints=num_waypoints,
        scale_components=scale_components,
        max_iterations=max_iterations,
        use_early_cell_as_start=use_early_cell_as_start,
        seed=seed,
    )

    pseudotime = pr_res.pseudotime.reindex(internal_index)
    adata.obs[f"{store_prefix}_pseudotime"] = pseudotime.to_numpy()

    entropy = getattr(pr_res, "entropy", None)
    if entropy is not None:
        entropy = entropy.reindex(internal_index)
        adata.obs[f"{store_prefix}_entropy"] = entropy.to_numpy()

    branch_probs = pr_res.branch_probs.reindex(internal_index)
    adata.obsm[f"{store_prefix}_branch_probs"] = branch_probs.to_numpy()
    adata.uns[f"{store_prefix}_branch_names"] = list(branch_probs.columns)

    adata.obsm[f"{store_prefix}_ms_data"] = ms_data.reindex(internal_index).to_numpy()
    adata.uns[f"{store_prefix}_ms_columns"] = list(ms_data.columns)

    adata.uns[f"{store_prefix}_cell_index_map"] = pd.DataFrame(
        {
            "internal_id": internal_index.astype(str),
            "obs_name": adata.obs_names.astype(str),
            "obs_pos": np.arange(adata.n_obs),
        }
    )

    terminal_states_out = getattr(pr_res, "terminal_states", None)
    if terminal_states_out is not None:
        adata.uns[f"{store_prefix}_terminal_states_internal"] = list(terminal_states_out)

    waypoints = getattr(pr_res, "waypoints", None)
    if waypoints is not None:
        adata.uns[f"{store_prefix}_waypoints_internal"] = list(waypoints)

    adata.uns[f"{store_prefix}_params"] = {
        "source": "X",
        "start_idx": int(start_idx),
        "early_cell_internal": str(early_cell),
        "n_diff_components": int(n_diff_components),
        "knn": int(knn),
        "num_waypoints": int(num_waypoints),
        "scale_components": bool(scale_components),
        "max_iterations": int(max_iterations),
        "use_early_cell_as_start": bool(use_early_cell_as_start),
        "seed": int(seed),
    }

    return adata


def resample_trajectory(
    adata: ad.AnnData,
    window_size: int = 100,
    step_size: int = 50,
    cell_dist_key: str = "X_umap",
    parameterization_key: str = "palantir_pseudotime",
    branch_prob_key: str = "palantir_branch_probs",
    store_key: str = "trajectory",
    min_cells_per_window: int = 5,
    **kwargs,
):
    if adata.n_obs < 1:
        raise ValueError("Trajectory resampling requires at least one observation")
    if cell_dist_key not in adata.obsm:
        raise KeyError(f"{cell_dist_key} not found in adata.obsm")
    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")
    if min_cells_per_window < 1:
        raise ValueError("min_cells_per_window must be at least 1")
    points = np.asarray(adata.obsm[cell_dist_key], dtype=float)
    time = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)
    if points.ndim != 2 or points.shape[0] != adata.n_obs:
        raise ValueError(f"{cell_dist_key} must be a 2D array with one row per observation")
    if not np.isfinite(time).all():
        raise ValueError(f"{parameterization_key} must contain only finite values")

    order = np.argsort(time)
    points_sorted = points[order]
    time_sorted = time[order]

    branch_prob = adata.obsm.get(branch_prob_key)
    if branch_prob is not None:
        branch_prob = np.asarray(branch_prob, dtype=float)
        if branch_prob.ndim == 1:
            branch_prob = branch_prob[:, None]
        if branch_prob.ndim != 2 or branch_prob.shape[0] != adata.n_obs:
            raise ValueError(f"{branch_prob_key} must have one row per observation")
        if np.any(branch_prob[np.isfinite(branch_prob)] < 0):
            raise ValueError(f"{branch_prob_key} cannot contain negative probabilities")
        branch_prob_sorted = branch_prob[order]
        n_branches = branch_prob_sorted.shape[1]
    else:
        branch_prob_sorted = np.ones((adata.n_obs, 1), dtype=float)
        n_branches = 1

    starts = _window_starts(adata.n_obs, window_size, step_size)
    effective_window = min(window_size, adata.n_obs)
    n_traj_points = len(starts)

    traj_points = np.full((n_branches, n_traj_points, points.shape[1]), np.nan, dtype=float)
    traj_time = np.full((n_branches, n_traj_points), np.nan, dtype=float)
    traj_counts = np.zeros((n_branches, n_traj_points), dtype=int)

    traj_weight_sum = np.full((n_branches, n_traj_points), np.nan, dtype=float)

    for i, start in enumerate(starts):
        end = min(start + effective_window, adata.n_obs)

        window_points = points_sorted[start:end]
        window_time = time_sorted[start:end]
        window_bp = branch_prob_sorted[start:end]
        if window_points.shape[0] < min_cells_per_window:
            continue

        for b in range(n_branches):
            w = window_bp[:, b].copy()

            valid = (
                np.isfinite(w) & np.isfinite(window_time) & np.isfinite(window_points).all(axis=1)
            )
            if valid.sum() < min_cells_per_window:
                continue

            w = w[valid]
            p = window_points[valid]
            t = window_time[valid]

            w_sum = w.sum()
            if w_sum <= 0:
                continue
            traj_weight_sum[b, i] = w_sum

            center = np.average(p, axis=0, weights=w)
            center_time = np.average(t, weights=w)

            traj_points[b, i, :] = center
            traj_time[b, i] = center_time
            traj_counts[b, i] = valid.sum()

    adata.uns[store_key] = traj_points
    adata.uns[f"{store_key}_metadata"] = {
        "time": traj_time,
        "counts": traj_counts,
        "weight_sum": traj_weight_sum,
        "window_starts": np.asarray(starts, dtype=int),
        "window_size": int(effective_window),
        "step_size": int(step_size),
    }
    return adata


def metabolic_velocity_field(
    adata: ad.AnnData,
    window_size: int = 100,
    step_size: int = 50,
    parameterization_key: str = "time",
):
    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")

    X = _dense_matrix(adata)

    t = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)

    n_obs, n_feat = X.shape
    if n_obs < 2:
        raise ValueError("Need at least 2 observations to compute velocity")
    if not np.isfinite(t).all() or not np.isfinite(X).all():
        raise ValueError("Data and parameterization values must be finite")

    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    order = np.argsort(t)
    X_sorted = X[order]
    t_sorted = t[order]

    starts = _window_starts(n_obs, window_size, step_size)
    effective_window = min(window_size, n_obs)

    n_windows = len(starts)

    state_centers = np.full((n_windows, n_feat), np.nan, dtype=float)
    velocity_vectors = np.full((n_windows, n_feat), np.nan, dtype=float)
    time_centers = np.full(n_windows, np.nan, dtype=float)
    speeds = np.full(n_windows, np.nan, dtype=float)
    counts = np.zeros(n_windows, dtype=int)

    for i, start in enumerate(starts):
        end = min(start + effective_window, n_obs)

        Xw = X_sorted[start:end]  # (k, m)
        tw = t_sorted[start:end]  # (k,)
        k = len(tw)

        if k < 2:
            continue

        t0 = tw.mean()
        dt = tw - t0

        denom = np.sum(dt**2)
        if denom <= 0:
            continue

        r0 = Xw.mean(axis=0)
        Xc = Xw - r0[None, :]
        drdt = (dt[:, None] * Xc).sum(axis=0) / denom

        state_centers[i] = r0
        velocity_vectors[i] = drdt
        time_centers[i] = t0
        speeds[i] = np.linalg.norm(drdt)
        counts[i] = k

    result = {
        "state_centers": state_centers,
        "velocity_field": velocity_vectors,
        "speeds": speeds,
        "time_centers": time_centers,
        "counts": counts,
        "window_starts": np.asarray(starts, dtype=int),
    }
    adata.uns["metabolic_velocity"] = result
    return adata


def metabolite_trends(
    adata: ad.AnnData,
    parameterization_key: str = "time",
    window_size: int = 100,
    step_size: int = 50,
    kernel_stat: str = "median",
    feature_name_key: str | None = None,
):

    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")

    X = _dense_matrix(adata)

    t = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)

    n_obs, n_feat = X.shape
    if n_obs < 2:
        raise ValueError("Need at least 2 observations")
    if not np.isfinite(t).all():
        raise ValueError(f"{parameterization_key} must contain only finite values")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    kernel_stat = kernel_stat.lower()
    if kernel_stat not in {"median", "mean", "sum"}:
        raise ValueError("kernel_stat must be one of {'median', 'mean', 'sum'}")

    # sort by parameterization
    order = np.argsort(t)
    X_sorted = X[order]
    t_sorted = t[order]

    # window starts
    starts = _window_starts(n_obs, window_size, step_size)
    effective_window = min(window_size, n_obs)

    n_windows = len(starts)

    pooled = np.full((n_windows, n_feat), np.nan, dtype=float)
    time_centers = np.full(n_windows, np.nan, dtype=float)
    counts = np.zeros(n_windows, dtype=int)

    # sliding-window pooling
    for i, start in enumerate(starts):
        end = min(start + effective_window, n_obs)
        Xw = X_sorted[start:end]
        tw = t_sorted[start:end]

        if Xw.shape[0] == 0:
            continue

        if kernel_stat == "median":
            pooled[i] = np.nanmedian(Xw, axis=0)
        elif kernel_stat == "mean":
            pooled[i] = np.nanmean(Xw, axis=0)
        elif kernel_stat == "sum":
            pooled[i] = np.nansum(Xw, axis=0)

        time_centers[i] = np.nanmean(tw)
        counts[i] = Xw.shape[0]

    # significance testing: metabolite pooled trend vs time
    rho = np.full(n_feat, np.nan, dtype=float)
    pval = np.full(n_feat, np.nan, dtype=float)

    valid_time = np.isfinite(time_centers)

    for j in range(n_feat):
        y = pooled[:, j]
        valid = valid_time & np.isfinite(y)

        # need at least 3 points and some variation
        if valid.sum() < 3:
            continue
        if np.nanstd(y[valid]) == 0:
            continue

        r, p = spearmanr(time_centers[valid], y[valid])
        rho[j] = r
        pval[j] = p

    # BH-FDR correction
    def _bh_fdr(p):
        p = np.asarray(p, dtype=float)
        q = np.full_like(p, np.nan)
        valid = np.isfinite(p)
        if valid.sum() == 0:
            return q

        pv = p[valid]
        m = len(pv)
        order = np.argsort(pv)
        ranked = pv[order]

        q_ranked = ranked * m / (np.arange(1, m + 1))
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0, 1)

        qv = np.empty_like(pv)
        qv[order] = q_ranked
        q[valid] = qv
        return q

    qval = _bh_fdr(pval)

    # feature names
    if feature_name_key is not None and feature_name_key in adata.var.columns:
        feature_names = np.asarray(adata.var[feature_name_key].astype(str))
    else:
        feature_names = np.asarray([f"feature_{i}" for i in range(n_feat)])

    # ranking: smallest q first, then larger |rho|
    rank_score = np.full(n_feat, np.inf, dtype=float)
    valid_test = np.isfinite(qval) & np.isfinite(rho)
    rank_score[valid_test] = qval[valid_test] - 1e-6 * np.abs(rho[valid_test])
    rank_idx = np.argsort(rank_score)

    result = {
        "pooled": pooled,  # (n_windows, n_feat)
        "time_centers": time_centers,  # (n_windows,)
        "counts": counts,  # (n_windows,)
        "rho": rho,  # Spearman correlation
        "pval": pval,
        "qval": qval,
        "feature_names": feature_names,
        "window_size": effective_window,
        "step_size": step_size,
        "kernel_stat": kernel_stat,
        "parameterization_key": parameterization_key,
        "sorted_obs_order": order,
        "rank_idx": rank_idx,
    }

    adata.uns["metabolite_trends"] = result
    return adata


def trend_cluster(
    trends: np.ndarray, metric: str = "correlation", cluster_method: str = "leiden", **kwargs
):
    trends = np.asarray(trends, dtype=float)
    if trends.ndim != 2:
        raise ValueError("trends must be a 2D array of shape (n_points, n_features)")

    _n_points, total_features = trends.shape
    if total_features < 2:
        return np.zeros(total_features, dtype=int)

    # feature-wise matrix: rows = features, cols = points
    X = trends.T.copy()  # shape = (n_features, n_points)

    # replace non-finite rows conservatively
    row_valid = np.isfinite(X).all(axis=1)
    if row_valid.sum() == 0:
        raise ValueError("No valid feature trends found")
    if row_valid.sum() == 1:
        out = np.full(total_features, -1, dtype=int)
        out[row_valid] = 0
        return out
    X = X[row_valid]
    n_features = X.shape[0]

    metric = metric.lower()
    cluster_method = cluster_method.lower()

    supported_metrics = {"cosine", "correlation", "euclidean", "sqeuclidean"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    supported_methods = {"leiden", "louvain", "agglomerative", "kmeans"}
    if cluster_method not in supported_methods:
        raise ValueError(f"Unsupported cluster_method: {cluster_method}")

    # ----------------------------
    # Helper: pairwise distance
    # ----------------------------
    if metric == "sqeuclidean":
        D = squareform(pdist(X, metric="euclidean")) ** 2
    else:
        D = squareform(pdist(X, metric=metric))

    # numerical cleanup
    D = np.asarray(D, dtype=float)
    D[~np.isfinite(D)] = np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 1.0
    np.fill_diagonal(D, 0.0)

    # ----------------------------
    # Graph-based clustering
    # ----------------------------
    if cluster_method in {"leiden", "louvain"}:
        try:
            import igraph as ig
        except ImportError as e:
            raise ImportError("igraph is required for leiden/louvain clustering") from e

        n_neighbors = int(kwargs.get("n_neighbors", 10))
        resolution = float(kwargs.get("resolution", 1.0))

        n_neighbors = max(1, min(n_neighbors, n_features - 1))

        # Build kNN graph from precomputed distance matrix
        # For each node, connect to nearest neighbors
        edges = set()
        weights = []

        for i in range(n_features):
            nn_idx = np.argsort(D[i])[1 : n_neighbors + 1]  # skip self
            for j in nn_idx:
                a, b = sorted((i, int(j)))
                edges.add((a, b))

        edges = sorted(edges)

        if len(edges) == 0:
            return np.zeros(n_features, dtype=int)

        # similarity from distance
        # robust scale from positive distances
        dvals = np.array([D[i, j] for i, j in edges], dtype=float)
        pos = dvals[dvals > 0]
        sigma = np.median(pos) if len(pos) > 0 else 1.0
        if sigma <= 0:
            sigma = 1.0

        weights = np.exp(-dvals / sigma).tolist()

        g = ig.Graph(n=n_features, edges=edges, directed=False)
        g.es["weight"] = weights

        if cluster_method == "leiden":
            try:
                import leidenalg
            except ImportError as e:
                raise ImportError("leidenalg is required for leiden clustering") from e

            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=g.es["weight"],
                resolution_parameter=resolution,
            )
            labels = np.array(part.membership, dtype=int)

        else:  # louvain
            part = g.community_multilevel(weights=g.es["weight"])
            labels = np.array(part.membership, dtype=int)

    # ----------------------------
    # Agglomerative clustering
    # ----------------------------
    elif cluster_method == "agglomerative":
        n_clusters = int(kwargs.get("n_clusters", 5))
        linkage = kwargs.get("linkage", "average")
        if not 1 <= n_clusters <= n_features:
            raise ValueError("n_clusters must be between 1 and the number of valid features")

        if linkage not in {"average", "complete", "single", "ward"}:
            raise ValueError("linkage must be one of {'average', 'complete', 'single', 'ward'}")

        if linkage == "ward":
            if metric not in {"euclidean", "sqeuclidean"}:
                raise ValueError("ward linkage requires euclidean-like metric")
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
            )
            labels = model.fit_predict(X)
        else:
            try:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage=linkage,
                )
            except TypeError:
                # older sklearn
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    linkage=linkage,
                )
            labels = model.fit_predict(D)

    # ----------------------------
    # KMeans
    # ----------------------------
    elif cluster_method == "kmeans":
        n_clusters = int(kwargs.get("n_clusters", 5))
        random_state = int(kwargs.get("random_state", 0))
        if not 1 <= n_clusters <= n_features:
            raise ValueError("n_clusters must be between 1 and the number of valid features")

        # optional normalization for shape-based clustering
        normalize = bool(kwargs.get("normalize", metric in {"cosine", "correlation"}))
        X_km = X.copy()

        if normalize:
            if metric == "correlation":
                X_km = X_km - X_km.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(X_km, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            X_km = X_km / norm

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        )
        labels = model.fit_predict(X_km)

    out = np.full(total_features, -1, dtype=int)
    out[row_valid] = labels
    return out
