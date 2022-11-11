import time
import re

from splink.duckdb.duckdb_linker import DuckDBLinker
import splink.duckdb.duckdb_comparison_library as cl

from path_fns.filepaths import (
    FINAL_CORRUPTED_OUTPUT_FILES_BASE,
    DEDUPE_OUTPUTS_FILES_BASE,
)

from synthetic_data.prepare import (
    prepare_df
)

from synthetic_data.random_match import (
    get_prob_two_rnd_recs_are_match
)

import pandas as pd
import os
from pathlib import Path

Path(DEDUPE_OUTPUTS_FILES_BASE).mkdir(parents=True, exist_ok=True)

link_settings_dir = os.path.join(
    DEDUPE_OUTPUTS_FILES_BASE,
    "settings"
)

Path(link_settings_dir).mkdir(parents=True, exist_ok=True)


for synthetic_data_path in Path(FINAL_CORRUPTED_OUTPUT_FILES_BASE).glob('*.parquet'):

    synthetic_data_filename = os.path.basename(
        synthetic_data_path
    )

    print(f"\nWorking on: {synthetic_data_filename}")
    start_time = time.time()

    match = re.search("^max_corruptions-(\d+)_prob_mult-(\d+\.?\d*)_set-(\d+)\.parquet$", synthetic_data_filename)

    max_dupes = int(match.group(1))
    corruption_probability_multiplier = float(match.group(2))
    set_id = int(match.group(3))

    link_settings_output = os.path.join(
        link_settings_dir,
        f"""max_corruptions-{max_dupes}_prob_mult-{corruption_probability_multiplier}_set-{set_id}.json"""
    )

    out_path = os.path.join(
        DEDUPE_OUTPUTS_FILES_BASE, 
        f"""max_corruptions-{max_dupes}_prob_mult-{corruption_probability_multiplier}_set-{set_id}.parquet"""
    )

    if os.path.exists(out_path):
        print("Output already exists. Skipping.")
        continue

    df_records = pd.read_parquet(synthetic_data_path)
    df_clean = prepare_df(df_records)

    distinct_entities = df_clean['cluster'].nunique()

    linker = DuckDBLinker(df_clean, connection=":temporary:")

    # linkage settings
    settings = {
        "probability_two_random_records_match": get_prob_two_rnd_recs_are_match(distinct_entities, max_dupes),
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": [
            "substr(l.given_name, 1, 2) = substr(r.given_name, 1, 2) and l.family_name = r.family_name",
            "substr(l.family_name, 1, 2) = substr(r.family_name, 1, 2) and l.given_name = r.given_name",
            "l.dob_d = r.dob_d and l.dob_m = r.dob_m and l.dob_y = r.dob_y and substr(l.given_name, 1, 2) = substr(r.given_name, 1, 2)",
            "l.dob_d = r.dob_d and l.dob_m = r.dob_m and l.dob_y = r.dob_y and substr(l.family_name, 1, 2) = substr(r.family_name, 1, 2)",
        ],
        "comparisons": [
            cl.jaro_winkler_at_thresholds(
                "given_name", [0.9, 0.7]
            ),
            cl.jaro_winkler_at_thresholds(
                "family_name", [0.9, 0.7]
            ),
            cl.exact_match("dob_d"),
            cl.exact_match("dob_m"),
            cl.exact_match("dob_y"),

            cl.exact_match("gender"),
        ],
        "retain_matching_columns": False,
        "retain_intermediate_calculation_columns": True,
        "additional_columns_to_retain": ["cluster"],
        "max_iterations": 10,
        "em_convergence": 0.01,
    }

    # Estimate m and u values
    linker.initialise_settings(settings)
    linker.estimate_u_using_random_sampling(target_rows=1e8)
    blocking_rule = "l.given_name = r.given_name and l.family_name = r.family_name"
    training_session_names = linker.estimate_parameters_using_expectation_maximisation(
        blocking_rule
    )
    blocking_rule = "l.dob_d = r.dob_d and l.dob_m = r.dob_m and l.dob_y = r.dob_y and substr(l.given_name, 1, 2) = substr(r.given_name, 1, 2)"
    training_session_dob1 = linker.estimate_parameters_using_expectation_maximisation(
        blocking_rule
    )
    blocking_rule = "l.dob_d = r.dob_d and l.dob_m = r.dob_m and l.dob_y = r.dob_y and substr(l.family_name, 1, 2) = substr(r.family_name, 1, 2)"
    training_session_dob1 = linker.estimate_parameters_using_expectation_maximisation(
        blocking_rule
    )

    linker.save_settings_to_json(link_settings_output, overwrite=True)

    df_predict = linker.predict()
    df_edges = df_predict.as_pandas_dataframe()

    df_edges = df_edges.drop(
        [
            "match_key",
        ],
        axis=1,
    )

    df_edges.to_parquet(out_path, index=False)

    duration = time.time() - start_time
    print(f"\nFinished. Took {duration:.2f}s\n")
