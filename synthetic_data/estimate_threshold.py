import pandas as pd
import numpy as np

def get_weight_corresponding_max_fmeasure(df, sample = None):
    if sample is not None:
        df = df.sample(n=sample, axis=0, ignore_index=True).copy()
    else:
        df = df.copy()
    
    match_weights = sorted(df.match_probability.unique())

    f_measures = []
    for match_weight in match_weights:
        true_positives = float(len(df.loc[(df["match_weight"] >= match_weight) & (df["cluster_l"] == df["cluster_r"])]))
        false_positives = float(len(df.loc[(df["match_weight"] >= match_weight) & (df["cluster_l"] != df["cluster_r"])]))
        false_negatives = float(len(df.loc[(df["match_weight"] < match_weight) & (df["cluster_l"] == df["cluster_r"])]))

        f_measures.append(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))

    max_fmeasure_index = f_measures.index(max(f_measures))

    return match_weights[max_fmeasure_index]