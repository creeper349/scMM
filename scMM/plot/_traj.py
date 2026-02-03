import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import umap
import logging
from pydiffmap import diffusion_map as dm

class PseudoTrajectory:
    """
    Scanpy-free pseudotime + trajectory inference

    Pipeline:
    High-D space (PCA / DiffMAP) -> kNN graph -> MST
                               -> longest path (backbone)
                               -> spline smoothing (global trajectory)
                               -> projection -> pseudotime
    """

    def __init__(self, X: np.ndarray, scale=True):
        self.X = np.asarray(X)
        self.scale = scale

        self.embedding_vis = None
        self.embedding_hd = None

        self.G = None
        self.G_mst = None

        self.backbones = []
        self.trajectory_curves = []

        self.pseudotime = None
        self.branch_id = None
        self.branch_nodes = None
        
    def embed(self,
              method="umap",
              n_components_vis=2,
              n_components_hd=8,
              **kwargs):

        X = self.X.copy()
        if self.scale:
            X = StandardScaler().fit_transform(X)
        logging.log(logging.INFO, f"Computing embeddings using method: {method}")

        if method == "pca":
            pca_hd = PCA(n_components=n_components_hd)
            self.embedding_hd = pca_hd.fit_transform(X)

            pca_vis = PCA(n_components=n_components_vis)
            self.embedding_vis = pca_vis.fit_transform(X)

        elif method == "umap":
            reducer_hd = umap.UMAP(n_components=n_components_hd, **kwargs)
            self.embedding_hd = reducer_hd.fit_transform(X)

            reducer_vis = umap.UMAP(n_components=n_components_vis, **kwargs)
            self.embedding_vis = reducer_vis.fit_transform(X)

        elif method == "diffmap":
            dmap_hd = dm.DiffusionMap.from_sklearn(
                n_evecs=n_components_hd,
                epsilon=kwargs.get("epsilon", "bgh"),
                k=kwargs.get("k", 64),
                alpha=kwargs.get("alpha", 0.5),
                oos=kwargs.get("oos", "dense")
            )
            self.embedding_hd = dmap_hd.fit_transform(X)

            self.embedding_vis = self.embedding_hd[:, :n_components_vis]

        else:
            raise ValueError(f"Unknown embedding method {method}")

        return self.embedding_vis, self.embedding_hd

    def compute_graph(self, n_neighbors=15):
       
        logging.log(logging.INFO, f"Computing kNN graph, n_neighbors = {n_neighbors}")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.embedding_hd)
        distances, indices = nbrs.kneighbors(self.embedding_hd)

        G = nx.Graph()
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    dist = np.linalg.norm(
                        self.embedding_hd[i] - self.embedding_hd[j]
                    )
                    G.add_edge(i, j, weight=dist)

        self.G = G
        return G

    def compute_mst(self):
        
        logging.log(logging.INFO, f"Computing minimum spanning tree")
        dist_mat = squareform(pdist(self.embedding_hd))
        mst = minimum_spanning_tree(dist_mat).toarray()
        self.G_mst = nx.from_numpy_array(mst + mst.T)
        return self.G_mst

    def detect_branches(self,
                        min_frac=0.3,
                        angle_thresh=40.0):
        """
        Branch points must satisfy:
        - degree > 2
        - at least one outgoing path is long enough
        - branching angle is large enough
        """
        logging.log(logging.INFO, f"Detecting branches")
        deg = dict(self.G_mst.degree())
        candidate = [n for n, d in deg.items() if d > 2]

        valid = []
        main_len = len(self.G_mst.nodes)

        for b in candidate:
            dirs = []
            long_branch = False

            for nb in self.G_mst.neighbors(b):
                path = nx.shortest_path(self.G_mst, b, nb)
                if len(path) > min_frac * main_len:
                    long_branch = True

                v = self.embedding_vis[path[1]] - self.embedding_vis[b]
                v = v / (np.linalg.norm(v) + 1e-8)
                dirs.append(v)

            if not long_branch or len(dirs) < 2:
                continue

            keep = False
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    angle = np.degrees(np.arccos(
                        np.clip(np.dot(dirs[i], dirs[j]), -1, 1)
                    ))
                    if angle > angle_thresh:
                        keep = True
                        break

            if keep:
                valid.append(b)

        self.branch_nodes = np.array(valid)
        return self.branch_nodes
    
    def find_backbones(self, start_idx=None):
        logging.log(logging.INFO, f"Construct backbones from cell {start_idx}")
        if start_idx is None:
            start_idx = 0

        leaves = [
            n for n, d in self.G_mst.degree()
            if d == 1 and n != start_idx
        ]

        backbones = []
        for leaf in leaves:
            path = nx.shortest_path(
                self.G_mst, start_idx, leaf
            )
            backbones.append(path)

        self.backbones = backbones
        return backbones
    
    def _resample_path(self, coords, n_samples=30):
        coords = np.asarray(coords)

        if len(coords) < 2:
            return coords

        seg = np.diff(coords, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        arc = np.insert(np.cumsum(seg_len), 0, 0.0)

        total = arc[-1]
        if total == 0:
            return coords

        arc /= total
        u_new = np.linspace(0, 1, n_samples)

        out = np.zeros((n_samples, coords.shape[1]))
        for d in range(coords.shape[1]):
            out[:, d] = np.interp(u_new, arc, coords[:, d])

        return out
    
    def _path_length(self, coords):
        if len(coords) < 2:
            return 0.0
        return np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()

    def fit_trajectory_curves(self,
                              n_points=300,
                              smooth=1.0,
                              min_branch_frac=0.2):

        curves = []
        logging.info("Fitting trajectory curves")

        lengths = []
        back_coords = []
        for backbone in self.backbones:
            coords = np.array([self.embedding_vis[i] for i in backbone])
            back_coords.append(coords)
            lengths.append(self._path_length(coords))

        max_len = max(lengths) if lengths else 0.0

        for backbone, coords, L in zip(self.backbones, back_coords, lengths):

            if max_len > 0 and L < min_branch_frac * max_len:
                continue

            coords = self._resample_path(coords, n_samples=40)

            _, uniq = np.unique(coords, axis=0, return_index=True)
            coords = coords[np.sort(uniq)]

            if len(coords) < 4:
                continue

            try:
                k = min(3, len(coords) - 1)
                tck, _ = splprep(coords.T, s=smooth, k=k)
                u_fine = np.linspace(0, 1, n_points)
                curve = np.vstack(splev(u_fine, tck)).T
            except Exception:
                curve = coords

            curves.append(curve)

        self.trajectory_curves = curves
        return curves

    def project_pseudotime(self):

        trees = []
        arcs = []
        for curve in self.trajectory_curves:
            seg = np.diff(curve, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            arc = np.insert(np.cumsum(seg_len), 0, 0.0)
            arc /= arc[-1]
            trees.append(cKDTree(curve))
            arcs.append(arc)

        pt = np.zeros(len(self.embedding_vis))
        branch_id = np.zeros(len(self.embedding_vis), dtype=int)

        for b, (tree, arc) in enumerate(zip(trees, arcs)):
            dists, idxs = tree.query(self.embedding_vis, k=1)
            mask = (pt == 0) | (dists < pt)
            pt[mask] = arc[idxs[mask]]
            branch_id[mask] = b

        self.pseudotime = pt
        self.branch_id = branch_id
        return pt, branch_id

    def run(self,
            embed_method="umap",
            start_idx=0,
            n_components_hd=15,
            n_components_vis=2,
            n_neighbors=15,
            **kwargs):

        self.embed(
            method=embed_method,
            n_components_hd=n_components_hd,
            n_components_vis=n_components_vis,
            **kwargs
        )

        self.compute_graph(n_neighbors=n_neighbors)
        self.compute_mst()

        self.detect_branches(kwargs.get("min_frac", 0.3), kwargs.get("angle_thresh", 40.0))
        self.find_backbones(start_idx=start_idx)

        self.fit_trajectory_curves(kwargs.get("n_points", 100),
                                   kwargs.get("smooth", 1.0),
                                   kwargs.get("min_branch_frac", 0.3))
        self.project_pseudotime()

        return self