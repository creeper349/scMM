library(TSCAN)
library(uwot)

default_params <- list(
  reduce_dim = list(
    n_pcs = 30,
    n_neighbors = 30,
    min_dist = 0.3
  )
)

get_param <- function(params, key, default) {
  if (!is.null(params[[key]])) return(params[[key]])
  default
}

make_embedding <- function(X, params=list()) {
    
  keep <- apply(X, 2, sd) > 0
  Xf <- X[, keep, drop=FALSE]

  rs <- sqrt(rowSums(Xf^2))
  rs[rs == 0] <- 1
  Xn <- Xf / rs

  npcs <- get_param(params, "n_pcs", 30)
  pca <- prcomp(Xn, rank. = npcs, center = TRUE, scale. = FALSE)

  list(
    pca = pca$x,
    umap = uwot::umap(
      pca$x,
      n_neighbors = get_param(params, "n_neighbors", 30),
      min_dist = get_param(params, "min_dist", 0.3),
      metric = "euclidean"
    )
  )
}

run_tscan <- function(X, obs, var, params=list()) {
  params <- modifyList(default_params, params)

  emb <- make_embedding(X, params$reduce_dim)

  clust <- exprmclust(emb$pca, reduce = FALSE)

  pst <- TSCANorder(clust, orderonly = TRUE)

  list(
    coordinates = emb$umap,
    pseudotime = pst,
    branch = clust$clusterid 
  )
}