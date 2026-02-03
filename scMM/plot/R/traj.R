suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 6){
  stop("Usage: Rscript traj.R input_X.csv input_obs.csv input_var.csv params.json method output_file_prefix pkg_dir")
}

input_X <- args[1]
input_obs <- args[2]
input_var <- args[3]
params_file <- args[4]
tool <- args[5]
output_prefix <- args[6]
pkg_dir <- args[7]

source(normalizePath(file.path(pkg_dir, "R", "slingshot.R"), mustWork = TRUE))
source(normalizePath(file.path(pkg_dir, "R", "dpt.R"), mustWork = TRUE))
source(normalizePath(file.path(pkg_dir, "R", "tscan.R"), mustWork = TRUE))

X <- fread(input_X, data.table = FALSE)
X <- X[ , -1, drop = FALSE]
X <- as.matrix(X)      

obs <- fread(input_obs, data.table = FALSE)
obs <- obs[ , -1, drop = FALSE] 

var <- fread(input_var, data.table = FALSE)
var <- var[ , -1, drop = FALSE]

params <- fromJSON(params_file)

res <- switch(tool,
              slingshot = run_slingshot(X, obs, var, params),
              dpt       = run_dpt(X, obs, params),
              tscan     = run_tscan(X, obs, params),
              stop(paste("Unsupported tool:", tool)))

stopifnot(all(c("coordinates", "pseudotime", "branch") %in% names(res)))

out <- list(
  metadata = list(
    method = tool,
    n_cells = nrow(res$coordinates),
    n_dim = ncol(res$coordinates),
    timestamp = Sys.time()
  ),
  cell_id = rownames(res$coordinates),
  coordinates = unname(as.matrix(res$coordinates)),
  pseudotime = as.numeric(res$pseudotime),
  branch = as.integer(res$branch)
)

json_path <- paste0(output_prefix, ".json.gz")
con <- gzfile(json_path, "wt")
writeLines(toJSON(out, auto_unbox = TRUE), con)
close(con)

message("Trajectory finished. JSON output: ", json_path)