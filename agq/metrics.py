from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import ndcg_score
from typing import Tuple, List, Dict


def _get_rank_pairs(truth_ranking: pd.DataFrame, estimated_ranking: pd.DataFrame, weighted: bool = False) -> Tuple[List, List]:
    """

    Parameters
    ----------
    truth_ranking
    estimated_ranking
    weighted

    Returns
    -------

    """
    max_rank, _ = estimated_ranking.shape
    truth_ranking = truth_ranking.iloc[:max_rank]

    truth_idx2rank = {v['idx']: rank for rank, v in truth_ranking.iterrows()}
    estimated_idx2rank = {v['idx']: rank for rank, v in estimated_ranking.iterrows()}

    assert len(estimated_idx2rank) == len(truth_idx2rank) and set(estimated_idx2rank.values()) == set(
        truth_idx2rank.values())

    if weighted is True:
        weights = np.linspace(0, 1, max_rank)[::-1] / max_rank
    else:
        weights = np.ones(max_rank) / max_rank

    locs, true_locs = [], []
    for idx, location in estimated_idx2rank.items():
        locs.append(location)
        true_locs.append(truth_idx2rank.get(idx, len(estimated_idx2rank) + 1))
    return locs, true_locs


def get_spearman(truth_ranking: pd.DataFrame, estimated_ranking: pd.DataFrame, weighted: bool = False) -> float:
    """

    Parameters
    ----------
    truth_ranking
    estimated_ranking
    weighted

    Returns
    -------

    """
    l, tl = _get_rank_pairs(truth_ranking, estimated_ranking, weighted=weighted)
    r = stats.spearmanr(l, tl)
    return r.correlation


def get_ndcg(truth_ranking: pd.DataFrame, estimated_ranking: pd.DataFrame, weighted: bool = False) -> float:
    """

    Parameters
    ----------
    truth_ranking
    estimated_ranking
    weighted

    Returns
    -------

    """
    l, tl = _get_rank_pairs(truth_ranking, estimated_ranking, weighted=weighted)
    true_relevance = [[len(tl) - i for i in tl]]
    scores = [[len(l) - j for j in l]]
    return ndcg_score(true_relevance, scores)


def get_rank_error(truth_ranking: pd.DataFrame, estimated_ranking: pd.DataFrame, weighted: bool = False) -> float:
    """

    Parameters
    ----------
    truth_ranking
    estimated_ranking
    weighted

    Returns
    -------

    """
    max_rank, _ = estimated_ranking.shape
    truth_ranking = truth_ranking.iloc[:max_rank]

    truth_idx2rank = {v['idx']: rank for rank, v in truth_ranking.iterrows()}
    estimated_idx2rank = {v['idx']: rank for rank, v in estimated_ranking.iterrows()}

    assert len(estimated_idx2rank) == len(truth_idx2rank) and set(estimated_idx2rank.values()) == set(
        truth_idx2rank.values())

    if weighted is True:
        weights = np.linspace(0, 1, max_rank)[::-1] / max_rank
    else:
        weights = np.ones(max_rank) / max_rank

    total_discrepancy = 0
    for idx, location in estimated_idx2rank.items():
        discrepancy = np.abs(location - truth_idx2rank.get(idx, len(estimated_idx2rank) + 1))
        total_discrepancy += weights[location] * discrepancy

    return total_discrepancy


def get_hit_ratio(truth_ranking: pd.DataFrame, estimated_ranking: pd.DataFrame) -> float:
    """

    Parameters
    ----------
    truth_ranking
    estimated_ranking

    Returns
    -------

    """
    max_rank, _ = estimated_ranking.shape
    truth_ranking = truth_ranking.iloc[:max_rank]

    truth_idx2rank = {v['idx']: rank for rank, v in truth_ranking.iterrows()}
    estimated_idx2rank = {v['idx']: rank for rank, v in estimated_ranking.iterrows()}

    assert len(estimated_idx2rank) == len(truth_idx2rank) and set(estimated_idx2rank.values()) == set(
        truth_idx2rank.values())

    return len(set(estimated_idx2rank.keys()).intersection(set(truth_idx2rank.keys()))) / max_rank


def evaluate(estimated: List[Dict], truth: List[Dict]) -> Dict:
    """

    Parameters
    ----------
    estimated: [{'idx': idx, 'P90': p90, 'count': count}]
    truth: [{'idx': idx, 'P90': p90, 'count': count}]

    Returns
    -------

    """
    estimated2idx = {e["idx"]: location for location, e in enumerate(estimated)}
    truth2idx = {e["idx"]: location for location, e in enumerate(truth)}

    scores = {
        "rank error": get_rank_error(truth2idx, estimated2idx, weighted=False),
        "rank error weighted": get_rank_error(truth2idx, estimated2idx, weighted=True),
        "hit ratio": get_hit_ratio(truth2idx, estimated2idx)
    }
    return scores


def compute_errors(estimated: List[Dict], truth: List[Dict], show: bool=True) -> Tuple[Counter, List]:
    """

    Parameters
    ----------
    estimated: [{'idx': idx, 'P90': p90, 'count': count}]
    truth: [{'idx': idx, 'P90': p90, 'count': count}]
    show: boolean for showing the plots

    Returns
    -------

    """
    assert len(estimated) > len(truth)

    errors = Counter()
    detailed_errors = []
    estimated = [e['idx'] for e in estimated]
    truth = [e['idx'] for e in truth]
    for location, e in enumerate(estimated):
        try:
            t = truth.index(e)
        except ValueError:
            t = len(truth) + 1  # best case scenario
        discrepancy = np.abs(location - t)
        errors[discrepancy] += 1
        # discrepancy, count (according to t-digest structure)
        detailed_errors.append([discrepancy, estimated[location]['count']])
    if show is True:
        plt.hist(errors);
        plt.title(f"Rank error distribution for ranking by P90, estimating top {len(estimated)}")
        plt.show()
        plt.scatter([p[0] for p in detailed_errors], [p[1] for p in detailed_errors]);
        plt.title("Rank error versus count");
        plt.xlabel('error');
        plt.ylabel('count')
        plt.show()
    return errors, detailed_errors
