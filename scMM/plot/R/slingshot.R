library(SingleCellExperiment)
library(slingshot)
library(uwot)
library(mclust)

default_params <- list(
  reduce_dim = list(method="UMAP", n_neighbors=30, min_dist=0.3),
  cluster = list(G=5)
)

get_param <- function(params, key, default) {
  if (!is.null(params[[key]])) return(params[[key]])
  default
}

run_slingshot <- function(X, obs, var, params=list()) {
  params <- modifyList(default_params, params)

  embedding <- uwot::umap(
    X,
    n_neighbors=get_param(params$reduce_dim, "n_neighbors", 30),
    min_dist=get_param(params$reduce_dim, "min_dist", 0.3),
    metric = "cosine"
  )

  clust <- mclust::Mclust(embedding, G=get_param(params$cluster, "G", 5))$classification

  sce <- SingleCellExperiment(list(counts=t(X)))
  reducedDims(sce) <- SimpleList(UMAP=embedding)

  sce <- SingleCellExperiment(
    assays = list(counts = t(X)),
    colData = obs,
    rowData = var
  )

  curves <- slingCurves(sce)
  pst <- slingPseudotime(sce)[,1]

  branch <- clust

  list(
    coordinates = embedding,
    pseudotime = pst,
    branch = branch
  )
}