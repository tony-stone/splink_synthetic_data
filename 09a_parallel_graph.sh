#!/bin/bash
set -e
set -o pipefail

for maxrecs in 99 #49
do
    for mult in 5 #0.5 1 2 5 
    do
        for id in {0..4}
        do
            $HOME/.local/bin/poetry run python 09_graph_measures.py --max_corr_recs=$maxrecs --global_prob_mult=$mult --set_id=$id
        done
    done
done

echo Initiated python scripts

