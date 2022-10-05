import argparse
from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np
import duckdb


from corrupt.corruption_functions import (
    master_record_no_op,
    format_master_data,
    generate_uncorrupted_output_record,
    format_master_record_first_array_item,
)

from corrupt.corrupt_name import (
    name_gen_uncorrupted_record,
    name_typo,
)


from corrupt.corrupt_date import (
    date_corrupt_timedelta,
    date_gen_uncorrupted_record,
)

from corrupt.corrupt_gender import (
    gender_gen_uncorrupted_record,
    gender_corrupt,
)

from path_fns.filepaths import (
    TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON,
    FINAL_CORRUPTED_OUTPUT_FILES_BASE,
)

from functools import partial
from corrupt.geco_corrupt import get_zipf_dist


from corrupt.record_corruptor import (
    CompositeCorruption,
    ProbabilityAdjustmentFromLookup,
    RecordCorruptor,
    ProbabilityAdjustmentFromSQL,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(message)s",
)
logger.setLevel(logging.INFO)


con = duckdb.connect()

in_path = TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON


# Configure how corruptions will be made for each field

# Col name is the OUTPUT column name.  For instance, we may input given name,
# family name etc to output full_name

# Guide to keys:
# format_master_data.  This functino may apply additional cleaning to the master
# record.  The same formatted master ata is then available to the
# 'gen_uncorrupted_record' and 'corruption_functions'


config = [
    {
        "col_name": "given_name",
        "format_master_data": partial(
            format_master_record_first_array_item, colname="given_nameLabel"
        ),
        "gen_uncorrupted_record": partial(
            name_gen_uncorrupted_record, input_colname="given_nameLabel", output_colname="given_name"
        ),
    },
    {
        "col_name": "family_name",
        "format_master_data": partial(
            format_master_record_first_array_item, colname="family_nameLabel"
        ),
        "gen_uncorrupted_record": partial(
            name_gen_uncorrupted_record, input_colname="family_nameLabel", output_colname="family_name"
        ),
    },
    {
        "col_name": "dob",
        "format_master_data": partial(
            format_master_record_first_array_item, colname="dob"
        ),
        "gen_uncorrupted_record": partial(
            date_gen_uncorrupted_record, input_colname="dob", output_colname="dob"
        ),
    },
    {
        "col_name": "gender",
        "format_master_data": partial(
            format_master_record_first_array_item, colname="sex_or_genderLabel"
        ),
        "gen_uncorrupted_record": gender_gen_uncorrupted_record,
    },
]


rc = RecordCorruptor()

# DOB
rc.add_simple_corruption(
    name="dob_timedelta",
    corruption_function=date_corrupt_timedelta,
    args={
        "input_colname": "dob",
        "output_colname": "dob",
        "num_days_delta": 50,
    },
    baseline_probability=0.1,
)

# Given name
rc.add_simple_corruption(
    name="given_name_typo",
    corruption_function=name_typo,
    args={
        "input_colname": "given_name", 
        "output_colname": "given_name"},
    baseline_probability=0.5,
)

# Family name
rc.add_simple_corruption(
    name="family_name_typo",
    corruption_function=name_typo,
    args={
        "input_colname": "family_name",
        "output_colname": "family_name"},
    baseline_probability=0.5,
)

# Gender
rc.add_simple_corruption(
    name="gender_corrupt",
    corruption_function=gender_corrupt,
    args={},
    baseline_probability=0.5,
)



max_corrupted_records = 20
zipf_dist = get_zipf_dist(max_corrupted_records)


Path(FINAL_CORRUPTED_OUTPUT_FILES_BASE).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="data corruption job runner")

    # parser.add_argument("--start_year", type=int)
    # parser.add_argument("--num_years", type=int)
    # args = parser.parse_args()
    # start_year = args.start_year
    # num_years = args.num_years

    # for year in range(start_year, start_year + num_years + 1):

    out_path = os.path.join(FINAL_CORRUPTED_OUTPUT_FILES_BASE, "all.parquet")

    if os.path.exists(out_path):
        exit

    sql = f"""
    select *
    from '{in_path}'
    LIMIT 5
    """

    raw_data = con.execute(sql).df()
    records = raw_data.to_dict(orient="records")

    output_records = []
    for i, master_input_record in enumerate(records):

        # Formats the input data into an easy format for producing
        # an uncorrupted/corrupted outputs records
        formatted_master_record = format_master_data(master_input_record, config)

        uncorrupted_output_record = generate_uncorrupted_output_record(
            formatted_master_record, config
        )
        uncorrupted_output_record["corruptions_applied"] = []

        output_records.append(uncorrupted_output_record)

        # How many corrupted records to generate
        total_num_corrupted_records = np.random.choice(
            zipf_dist["vals"], p=zipf_dist["weights"]
        )

        for i in range(total_num_corrupted_records):
            record_to_modify = uncorrupted_output_record.copy()
            record_to_modify["corruptions_applied"] = []
            record_to_modify["id"] = (
                uncorrupted_output_record["cluster"] + f"_{i+1}"
            )
            record_to_modify["uncorrupted_record"] = False
            rc.apply_probability_adjustments(uncorrupted_output_record)
            corrupted_record = rc.apply_corruptions_to_record(
                formatted_master_record,
                record_to_modify,
            )
            output_records.append(corrupted_record)

    df = pd.DataFrame(output_records)

    print(df)

    df.to_parquet(out_path, index=False)
    print(f"written file with {len(df):,.0f} records")
