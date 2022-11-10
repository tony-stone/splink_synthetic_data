import duckdb
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
from path_fns.filepaths import (
    TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON_DIR,
    TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON,
    PERSONS_PROCESSED_ONE_ROW_PER_PERSON,
    TRANSFORMED_MASTER_DATA_COUNT,
)


from transform_master_data.full_name_alternatives_per_person import (
    add_full_name_alternatives_per_person,
)
from transform_master_data.pipeline import SQLPipeline

from transform_master_data.parse_point import parse_point_to_lat_lng
from transform_master_data.to_lowercase import to_lowercase

Path(TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON_DIR).mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
pipeline = SQLPipeline(con)


sql = f"""
select *
from '{PERSONS_PROCESSED_ONE_ROW_PER_PERSON}'
where array_length(given_nameLabel) > 0
and array_length(family_nameLabel) > 0
and array_length(sex_or_genderLabel) = 1
and list_extract(sex_or_genderLabel, 1) IN ('male', 'female')
and array_length(dob) > 0
"""
# list_contains(country_citizen, 'Q145') and # Citizen of UK

pipeline.enqueue_sql(sql, "df")

pipeline = add_full_name_alternatives_per_person(
    pipeline, output_table_name="df_full_names", input_table_name="df"
)

pipeline = parse_point_to_lat_lng(
    pipeline,
    "birth_coordinates",
    output_table_name="df_bc_fixed",
    input_table_name="df_full_names",
)
pipeline = parse_point_to_lat_lng(
    pipeline,
    "residence_coordinates",
    output_table_name="df_rc_fixed",
    input_table_name="df_bc_fixed",
)

pipeline = to_lowercase(
    pipeline,
    "humanLabel",
    output_table_name="df_hl_fixed",
    input_table_name="df_rc_fixed",
)
pipeline = to_lowercase(
    pipeline,
    "humanAltLabel",
    output_table_name="df_hal_fixed",
    input_table_name="df_hl_fixed",
)
pipeline = to_lowercase(
    pipeline,
    "birth_name",
    output_table_name="df_bn_fixed",
    input_table_name="df_hal_fixed",
)
pipeline = to_lowercase(
    pipeline,
    "given_nameLabel",
    output_table_name="df_gnl_fixed",
    input_table_name="df_bn_fixed",
)
pipeline = to_lowercase(
    pipeline,
    "family_nameLabel",
    output_table_name="df_fnl_fixed",
    input_table_name="df_gnl_fixed",
)
pipeline = to_lowercase(
    pipeline,
    "full_name_arr",
    output_table_name="df_fna_fixed",
    input_table_name="df_fnl_fixed",
)

df = pipeline.execute_pipeline()


df_arrow = df.fetch_arrow_table()
df_pandas = df_arrow.to_pandas()

# ensure they represent distinct Wikidata entries
assert max(df_pandas["human"].value_counts()) == 1

# Drop Wikidata ID
df_pandas.drop("human", axis=1)

# Reset index
df_pandas.reset_index(drop=True, inplace=True)

# Attach new ID
df_pandas["human"] = df_pandas.index

index_length = len(df_pandas.index)
rng = np.random.default_rng()
df_pandas["group_id"] = rng.choice(
    index_length, 
    index_length, 
    replace=False
    ) % 5

df_pandas.to_parquet(TRANSFORMED_MASTER_DATA_ONE_ROW_PER_PERSON, index=False)
