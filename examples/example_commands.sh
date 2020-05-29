#!/usr/bin/env bash

################################################################################
#
#   Set directories.
#
################################################################################

scripts=../src
data=data
results=results

################################################################################
#
#   Find anomalies.
#
################################################################################

echo "Finding anomalies..."


# Find submatrix anomaly.
python3 $scripts/find_anomaly.py \
    -i $data/submatrix_scores.hdf5 \
    -a submatrix \
    -o $results/submatrix_results.tsv 

# Find line anomaly.
python3 $scripts/find_anomaly.py \
    -i $data/line_scores.tsv \
    -a line \
    -o $results/line_results.tsv

# Find cutsize anomaly.
python3 $scripts/find_anomaly.py \
    -i $data/cutsize_scores.tsv \
    -a cutsize \
    -o $results/cutsize_results.tsv \
    -elf $data/cutsize_graph.tsv -r 20
    
# Find unconstrained anomaly.
python3 $scripts/find_anomaly.py \
    -i $data/cutsize_scores.tsv \
    -a unconstrained \
    -o $results/unconstrained_results.tsv