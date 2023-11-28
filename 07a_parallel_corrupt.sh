#!/bin/bash
set -e
set -o pipefail

for maxrecs in 49 99
do
    for mult in 1 0.5 2 5
    do
        for id in {0..4}
        do
            $HOME/.local/bin/poetry run python 07_corrupt_records.py --max_corr_recs=$maxrecs --global_prob_mult=$mult --set_id=$id &
        done
    done
done

echo Finished Successfully

