import argparse
import time
import sys

from synthetic_data.estimate_threshold import (
    get_weight_corresponding_max_fmeasure
)

from graphs.linkagegraph import LinkageGraph

from path_fns.filepaths import (
    DEDUPE_OUTPUTS_FILES_BASE,
    GRAPH_OUTPUTS_BASE,
)

import pandas as pd
import os
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="data corruption job runner")

    parser.add_argument("--max_corr_recs", type=int)
    parser.add_argument("--global_prob_mult", type=float)
    parser.add_argument("--set_id", type=int)
    args = parser.parse_args()
    max_dupes = args.max_corr_recs
    corruption_probability_multiplier = args.global_prob_mult
    set_id = args.set_id

    Path(GRAPH_OUTPUTS_BASE).mkdir(parents=True, exist_ok=True)

    dedupe_output_path = os.path.join(DEDUPE_OUTPUTS_FILES_BASE, 
        f"""max_corruptions-{max_dupes}_prob_mult-{corruption_probability_multiplier}_set-{set_id}.parquet"""
    )

    out_path = os.path.join(
        GRAPH_OUTPUTS_BASE, 
        f"""max_corruptions-{max_dupes}_prob_mult-{corruption_probability_multiplier}_set-{set_id}.parquet"""
    )

    dedupe_output_filename = os.path.basename(
        dedupe_output_path
    )

    if not os.path.exists(dedupe_output_path):
        print(f"""File '{dedupe_output_path}' does not exist. Skipping.""")
        sys.exit(0)

    if os.path.exists(out_path):
        print(f"""File '{dedupe_output_filename}' already exists. Skipping.""")
        sys.exit(0)

    print(f"\nWorking on: {dedupe_output_filename}")
    start_time = time.time()

    dedupe_records = pd.read_parquet(dedupe_output_path)
    print(f"\nFile '{dedupe_output_filename}': read.")

    threshold = get_weight_corresponding_max_fmeasure(dedupe_records, int(len(dedupe_records)/50))
    print(f"\nFile '{dedupe_output_filename}': Threshold calculated.")

    graphs_all = LinkageGraph(dedupe_records, prob_threshold = threshold, min_order = 2)
    print(f"\nFile '{dedupe_output_filename}': Graph created.")

    df_gmeasures = graphs_all.get_measures()
    print(f"\nFile '{dedupe_output_filename}': Measures calculated.")

    df_gmeasures.to_parquet(out_path, index=False)

    duration = time.time() - start_time
    print(f"\nFile '{dedupe_output_filename}': Finished.\nTook {duration:.2f}s\n")
