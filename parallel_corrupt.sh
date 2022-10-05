#!/bin/zsh
set -e
set -o pipefail

for maxrecs in 50 100
do
    for mult in 1 0.5 2 5
    do
        for id in {1..10}
        do
            C:/Users/cm1avs/AppData/Local/pypoetry/Cache/virtualenvs/splink-synthetic-data-wgdmJF1G-py3.10/Scripts/python.exe c:/Users/cm1avs/dev/splink_synthetic_data/07_corrupt_records.py --max_corr_recs=$maxrecs --global_prob_mult=$mult --set_id=$id &
        done
    done
done

echo Finished Successfully

