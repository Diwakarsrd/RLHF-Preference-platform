"""
Metrics Engine for RLHF Platform.

Computes:
  - Inter-Annotator Agreement (IAA) - raw percentage
  - Cohen's Kappa (multi-annotator averaged pairwise)
  - Preference distribution
  - Annotator consistency (self-agreement on repeated pairs)
  - Confidence-weighted agreement
"""

from collections import defaultdict, Counter
from itertools import combinations
from typing import List, Dict, Tuple, Optional
import math


# 
# Core Agreement Calculations
# 

def raw_agreement(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Pairwise observed agreement (P_o).
    Returns value in [0, 1].
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return matches / len(labels_a)


def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Cohen's Kappa between two annotators.

    kappa = (P_o - P_e) / (1 - P_e)

    Interpretation:
      < 0.0   Poor
      0.0-0.2 Slight
      0.2-0.4 Fair
      0.4-0.6 Moderate
      0.6-0.8 Substantial
      0.8-1.0 Almost perfect
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    n = len(labels_a)
    categories = list(set(labels_a + labels_b))

    # Observed agreement
    p_o = raw_agreement(labels_a, labels_b)

    # Expected agreement
    p_e = 0.0
    for cat in categories:
        p_a = labels_a.count(cat) / n
        p_b = labels_b.count(cat) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0  # Perfect expected -> avoid division by zero

    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


def multi_annotator_kappa(annotations_by_pair: Dict[int, List[Tuple[str, str]]]) -> float:
    """
    Average pairwise Cohen's Kappa across all annotator combinations.

    annotations_by_pair: {pair_id: [(annotator_id, preference), ...]}

    Returns average kappa.
    """
    # Restructure: {annotator_id: {pair_id: preference}}
    annotator_labels: Dict[str, Dict[int, str]] = defaultdict(dict)
    for pair_id, ann_list in annotations_by_pair.items():
        for annotator_id, preference in ann_list:
            annotator_labels[annotator_id][pair_id] = preference

    annotators = list(annotator_labels.keys())
    if len(annotators) < 2:
        return 0.0

    kappas = []
    for ann_a, ann_b in combinations(annotators, 2):
        # Find pairs both annotated
        shared_pairs = set(annotator_labels[ann_a].keys()) & set(annotator_labels[ann_b].keys())
        if len(shared_pairs) < 2:
            continue
        labels_a = [annotator_labels[ann_a][p] for p in shared_pairs]
        labels_b = [annotator_labels[ann_b][p] for p in shared_pairs]
        kappas.append(cohens_kappa(labels_a, labels_b))

    return round(sum(kappas) / len(kappas), 4) if kappas else 0.0


# 
# Preference Distribution
# 

def preference_distribution(preferences: List[str]) -> Dict[str, float]:
    """Returns % for A, B, tie."""
    if not preferences:
        return {"A": 0.0, "B": 0.0, "tie": 0.0}
    n = len(preferences)
    counts = Counter(preferences)
    return {
        "A":   round(counts.get("A", 0)   / n * 100, 2),
        "B":   round(counts.get("B", 0)   / n * 100, 2),
        "tie": round(counts.get("tie", 0) / n * 100, 2),
    }


# 
# Consistency (self-agreement on repeated items)
# 

def annotator_consistency(
    annotations: List[Dict]  # list of {pair_id, annotator_id, preference}
) -> Dict[str, float]:
    """
    For each annotator who labeled the same pair multiple times,
    compute the fraction of consistent (matching) judgments.
    """
    # Group by annotator -> pair
    grouped: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        grouped[ann["annotator_id"]][ann["pair_id"]].append(ann["preference"])

    consistency: Dict[str, float] = {}
    for annotator, pair_votes in grouped.items():
        repeated = {p: votes for p, votes in pair_votes.items() if len(votes) > 1}
        if not repeated:
            continue
        # Fraction of repeated items where all votes agree
        consistent = sum(1 for votes in repeated.values() if len(set(votes)) == 1)
        consistency[annotator] = round(consistent / len(repeated), 4)

    return consistency


# 
# Confidence-Weighted Agreement
# 

def confidence_weighted_agreement(
    annotations_by_pair: Dict[int, List[Tuple[str, str, int]]]
    # {pair_id: [(annotator_id, preference, confidence), ...]}
) -> float:
    """
    Weighted agreement: high-confidence annotators count more.
    """
    weighted_agree = 0.0
    total_weight   = 0.0

    for pair_id, ann_list in annotations_by_pair.items():
        if len(ann_list) < 2:
            continue
        for (ann_a, pref_a, conf_a), (ann_b, pref_b, conf_b) in combinations(ann_list, 2):
            weight = (conf_a + conf_b) / 2
            if pref_a == pref_b:
                weighted_agree += weight
            total_weight += weight

    if total_weight == 0:
        return 0.0
    return round(weighted_agree / total_weight, 4)


# 
# Full Metrics Summary
# 

def compute_full_metrics(annotations: List[Dict]) -> Dict:
    """
    Main entry point.  Pass in a list of annotation dicts from the DB.

    Returns a metrics summary dict ready for the API / UI.
    """
    if not annotations:
        return {
            "total_annotations": 0,
            "unique_pairs":      0,
            "unique_annotators": 0,
            "preference_distribution": {"A": 0, "B": 0, "tie": 0},
            "raw_iaa":           None,
            "cohens_kappa":      None,
            "kappa_interpretation": "N/A - need >=2 annotators on shared pairs",
            "consistency":       {},
            "confidence_weighted_agreement": None,
        }

    preferences   = [a["preference"] for a in annotations]
    pair_ids      = set(a["pair_id"] for a in annotations)
    annotator_ids = set(a["annotator_id"] for a in annotations)

    # Build pair -> annotator mappings
    by_pair_simple: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    by_pair_conf:   Dict[int, List[Tuple[str, str, int]]] = defaultdict(list)
    for a in annotations:
        by_pair_simple[a["pair_id"]].append((a["annotator_id"], a["preference"]))
        by_pair_conf[a["pair_id"]].append(
            (a["annotator_id"], a["preference"], a.get("confidence", 3))
        )

    # Raw IAA - pairs with >=2 annotations
    multi_annotated = {p: v for p, v in by_pair_simple.items() if len(v) >= 2}
    raw_iaa = None
    if multi_annotated:
        agree_count = 0
        total_pairs = 0
        for pair_id, ann_list in multi_annotated.items():
            for (_, pref_a), (_, pref_b) in combinations(ann_list, 2):
                total_pairs += 1
                if pref_a == pref_b:
                    agree_count += 1
        raw_iaa = round(agree_count / total_pairs, 4) if total_pairs else 0.0

    kappa = multi_annotator_kappa(by_pair_simple)
    kappa_interp = interpret_kappa(kappa) if kappa is not None else "N/A"
    cwa = confidence_weighted_agreement(by_pair_conf)
    consistency = annotator_consistency(annotations)

    return {
        "total_annotations":   len(annotations),
        "unique_pairs":        len(pair_ids),
        "unique_annotators":   len(annotator_ids),
        "preference_distribution": preference_distribution(preferences),
        "raw_iaa":             raw_iaa,
        "cohens_kappa":        kappa,
        "kappa_interpretation": kappa_interp,
        "consistency":         consistency,
        "confidence_weighted_agreement": cwa,
    }


def interpret_kappa(k: float) -> str:
    if k < 0:
        return "Poor (< 0)"
    elif k < 0.2:
        return "Slight (0.0 - 0.2)"
    elif k < 0.4:
        return "Fair (0.2 - 0.4)"
    elif k < 0.6:
        return "Moderate (0.4 - 0.6)"
    elif k < 0.8:
        return "Substantial (0.6 - 0.8)"
    else:
        return "Almost Perfect (0.8 - 1.0)"
