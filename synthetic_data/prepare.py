import pandas as pd
import numpy as np

def prepare_df(df):
    """Prepare data frame for linkage

    Args:
        df (pandas dataframe): Data frame with st least the following columns: [
        "id",
        "cluster",
        "given_name",
        "family_name",
        "dob",
        "gender",
    ]

    Returns:
        pandas dataframe: Cleaned dataframe with (only) the following columns: [
        "id",
        "cluster",
        "given_name",
        "family_name",
        "dob_d",
        "dob_m",
        "dob_y",
        "gender",
    ]
    """

    cols_in = [
        "id",
        "cluster",
        "given_name",
        "family_name",
        "dob",
        "gender",
    ]

    df = df[cols_in].copy()

    for col in [
        "given_name",
        "family_name",
        "dob",
        "gender",
    ]:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": None})
        if df[col].isnull().sum() > 0:
            raise Exception(f"Nulls present in '{col}' field.") 

    df["dob_split"] = df["dob"].str.strip().str.split("-")
    df["dob_d"] = df["dob_split"].str[0]
    df["dob_m"] = df["dob_split"].str[1]
    df["dob_y"] = df["dob_split"].str[2]

    for col in [
        "dob_d",
        "dob_m",
        "dob_y",
    ]:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(r'^\s*$', None, regex=True)
        df[col] = df[col].fillna(np.nan).replace([np.nan, pd.NA], [None, None])
        if df[col].isnull().sum() > 0:
            raise Exception(f"Nulls present in '{col}' field.") 

    df.rename({'id': 'unique_id'}, axis=1, inplace=True)

    cols_out = [
        "unique_id",
        "cluster",
        "given_name",
        "family_name",
        "dob_d",
        "dob_m",
        "dob_y",
        "gender",
    ]

    return df[cols_out]
