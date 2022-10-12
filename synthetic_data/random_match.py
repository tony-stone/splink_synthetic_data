import numpy as np
from corrupt.geco_corrupt import get_zipf_dist

def get_prob_two_rnd_recs_are_match(distinct_entities: int, max_dupes: int) -> str:
    """Calculates the probability of two randomly drawn records belonging to
    the same entity given the number of entities and the maximum number of
    duplicates made of these entities (using a Zipf distribution)

    Args:
        distinct_entities (int): Number of distinct records
        max_dupes (int): Maximum number of duplicates

    Returns:
        float: A string giving the probability in the form "x in y".
    """
    zipf_dist = get_zipf_dist(max_dupes)
    zipf_dist_val = np.asarray(zipf_dist["vals"]).astype(np.float64)
    zipf_dist_p = np.asarray(zipf_dist["weights"])

    total_recs = float(round(np.sum((zipf_dist_val + 1) * np.round(zipf_dist_p * distinct_entities))))

    sum = float(round(
        distinct_entities
        / (total_recs - 1)
        * np.sum((zipf_dist_val + 1) * zipf_dist_p * zipf_dist_val))
    )

    return sum / total_recs
