#!/usr/bin/env Rscript
# SCART internal script — do not call directly.
# Called by preprocessing.py via subprocess.
# Args: input_rds output_csv sam_name genome id_type ngene_chr
#       win_size ks_cut distance n_cores plot_genes output_seg

args <- commandArgs(trailingOnly = TRUE)

input_rds   <- args[1]   # RDS file: list(mat, gene_names, barcodes, norm_cells)
output_csv  <- args[2]   # Where to write prediction CSV
sam_name    <- args[3]
genome      <- args[4]
id_type     <- args[5]
ngene_chr   <- as.integer(args[6])
win_size    <- as.integer(args[7])
ks_cut      <- as.numeric(args[8])
distance    <- args[9]
n_cores     <- as.integer(args[10])
plot_genes  <- as.logical(args[11])
output_seg  <- as.logical(args[12])

suppressPackageStartupMessages(library(copykat))

payload     <- readRDS(input_rds)
r_mat       <- payload$mat
norm_cells  <- payload$norm_cells

cat("[run_copykat.R] Matrix dims:", nrow(r_mat), "genes x", ncol(r_mat), "cells\n")
cat("[run_copykat.R] Normal reference cells:", length(norm_cells), "\n")

result <- copykat(
    rawmat          = r_mat,
    id.type         = id_type,
    ngene.chr       = ngene_chr,
    win.size        = win_size,
    KS.cut          = ks_cut,
    sam.name        = sam_name,
    distance        = distance,
    norm.cell.names = norm_cells,
    output.seg      = output_seg,
    plot.genes      = plot_genes,
    genome          = genome,
    n.cores         = n_cores
)

pred <- result$prediction
cat("[run_copykat.R] Prediction counts:\n")
print(table(pred$copykat.pred))

write.csv(pred, output_csv, row.names = FALSE, quote = FALSE)
cat("[run_copykat.R] Saved predictions to:", output_csv, "\n")
