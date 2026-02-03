library(diffusionMap)
library(igraph)
library(uwot)

default_params <- list(
  k = 30,
  root = 1
)

run_dpt <- function(X, obs, var, params=list()) {
  params <- modifyList(default_params, params)

  dm <- diffuse(X, k=get_param(params, "k", 30))
  coords <- as.matrix(dm$X[,1:2])

  g <- make_knn_graph(coords, k=get_param(params, "k", 30))

  d <- distances(g, v=get_param(params, "root", 1))
  pst <- as.numeric(d)

  list(
    coordinates = coords,
    pseudotime = pst,
    branch = rep(1, nrow(X))
  )
}

make_knn_graph <- function(X, k=30) {
  dmat <- as.matrix(dist(X))
  edges <- which(apply(dmat, 1, order)[1:k,], arr.ind=TRUE)
  graph_from_edgelist(cbind(edges[,2], edges[,1]), directed=FALSE)
}