#!/usr/bin/env Rscript
# Apply ashr's adaptive shrinkage to one chunk CSV produced by ComputeSE.py.
#
# Input  : chunk_XXXXXX.csv  with columns mean_diff, se (+ pert/gene/n metadata)
# Output : AshrResult_chunk_XXXXXX.csv  (sibling file)  with ashr's full result
#          frame (PosteriorMean, PosteriorSD, lfsr, NegativeProb, ...)
#
# Usage:  Rscript run_ashr_on_chunk.R chunk_000000.csv

suppressPackageStartupMessages(library(ashr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: run_ashr_on_chunk.R chunk_XXXXXX.csv")

infile  <- args[1]
outfile <- sub("chunk_", "AshrResult_chunk_", infile)

a <- read.csv(infile)
myRes <- ash(a$mean_diff, a$se)
write.csv(myRes$result, outfile, row.names = FALSE)
