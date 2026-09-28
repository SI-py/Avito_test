"""Leakage-safe experiments with hybrid candidate retrieval.

This script intentionally does not write submission files.  It uses clicked
train rows whose items are present in benchmark_items as pseudo-test labels.
All behavioural profiles are learned only from train rows whose items are NOT
in benchmark_items, so the relevant item itself cannot leak into a profile.
"""

from __future__ import annotations

from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


DATA = Path("data")
OUT = Path("hybrid_neural_results.json")
DENSE = Path("kaggle_output_v3/ml_embed_dense_top200.npz")
SEED = 20260927
N_GROUPS = 3200
RETRIEVAL_K = 180
TOP_K = 50


def normalize(value) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").map(normalize)


def sparse_topk(qmat, imat, k=RETRIEVAL_K, batch=64):
    """Return per-query item indices and cosine scores."""
    all_idx, all_val = [], []
    transposed = imat.T.tocsc()
    for start in range(0, qmat.shape[0], batch):
        scores = (qmat[start : start + batch] @ transposed).tocsr()
        for row in range(scores.shape[0]):
            left, right = scores.indptr[row], scores.indptr[row + 1]
            idx = scores.indices[left:right]
            val = scores.data[left:right]
            if len(val) > k:
                keep = np.argpartition(val, -k)[-k:]
                keep = keep[np.argsort(val[keep])[::-1]]
                idx, val = idx[keep], val[keep]
            elif len(val):
                order = np.argsort(val)[::-1]
                idx, val = idx[order], val[order]
            all_idx.append(idx.astype(np.int32, copy=False))
            all_val.append(val.astype(np.float32, copy=False))
    return all_idx, all_val


def rrf_score(channels, weights, rrf_k=60):
    result = defaultdict(float)
    for ranking, weight in zip(channels, weights):
        for rank, idx in enumerate(ranking):
            result[int(idx)] += weight / (rrf_k + rank + 1)
    return result


def rank_from_scores(scores, limit=TOP_K):
    return [idx for idx, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]]


def recall(ranking, relevant):
    return len(set(ranking[:TOP_K]) & relevant) / len(relevant)


def add_structured(
    scores,
    row,
    normalized_query,
    item_locs,
    item_microcats,
    transition_probs,
    exact_micro_probs,
    centroid_scores,
    popularity,
    exact_loc_bonus=0.0,
    transition_bonus=0.0,
    exact_micro_bonus=0.0,
    centroid_micro_bonus=0.0,
    popularity_bonus=0.0,
):
    source_loc = int(row["search_location_id"])
    transitions = transition_probs.get(source_loc, {})
    exact_micro = exact_micro_probs.get(normalized_query, {})
    centroid_micro = centroid_scores
    for idx in scores:
        loc = int(item_locs[idx])
        microcat = int(item_microcats[idx])
        if loc == source_loc:
            scores[idx] += exact_loc_bonus
        if transition_bonus:
            scores[idx] += transition_bonus * transitions.get(loc, 0.0)
        if exact_micro_bonus:
            scores[idx] += exact_micro_bonus * exact_micro.get(microcat, 0.0)
        if centroid_micro_bonus:
            scores[idx] += centroid_micro_bonus * centroid_micro.get(microcat, 0.0)
        if popularity_bonus:
            scores[idx] += popularity_bonus * popularity[idx]
    return scores


def quota_ranking(base_ranking, item_locs, query_loc, local_quota):
    local = [idx for idx in base_ranking if item_locs[idx] == query_loc]
    chosen = local[:local_quota]
    used = set(chosen)
    for idx in base_ranking:
        if idx not in used:
            chosen.append(idx)
            used.add(idx)
        if len(chosen) == TOP_K:
            break
    return chosen


started = time.time()
print("Loading parquet files", flush=True)
train_columns = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category", "item_id",
    "item_microcat_id", "item_location_id",
]
train = pd.read_parquet(DATA / "train.parquet", columns=train_columns)
items = pd.read_parquet(DATA / "benchmark_items.parquet")
item_ids = items["item_id"].astype(str).to_numpy()
item_id_to_idx = {value: idx for idx, value in enumerate(item_ids)}
benchmark_item_ids = set(item_ids)
item_locs = items["item_location_id"].to_numpy()
item_microcats = items["item_microcat_id"].to_numpy()

# Validation labels: exact query feature tuples, with all clicked benchmark
# items for the tuple treated as relevant.
query_cols = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
val_rows = train.loc[train["item_id"].isin(benchmark_item_ids), query_cols + ["item_id"]].copy()
val_rows["_group"] = pd.util.hash_pandas_object(val_rows[query_cols].fillna(""), index=False).astype(str)
relevant_ids = val_rows.groupby("_group")["item_id"].agg(set)
validation = val_rows.drop_duplicates("_group").set_index("_group")
validation["relevant"] = relevant_ids
validation = validation.sample(min(N_GROUPS, len(validation)), random_state=SEED)
validation["relevant_idx"] = validation["relevant"].map(
    lambda values: {item_id_to_idx[value] for value in values}
)
split = len(validation) // 2
tune_positions = np.arange(split)
test_positions = np.arange(split, len(validation))

val_main = clean(validation["search_query"])
val_params = clean(validation["search_infm_params_text"])
val_mixed_text = val_main + " " + val_main + " " + val_params

# Leakage-safe historical rows.  These can transfer intent and geography but
# contain none of the validation corpus items.
profile = train.loc[
    ~train["item_id"].isin(benchmark_item_ids),
    ["search_query", "search_location_id", "item_location_id", "item_microcat_id"],
].copy()
profile["normalized_query"] = clean(profile["search_query"])

loc_counts = (
    profile.groupby(["search_location_id", "item_location_id"])
    .size()
    .rename("n")
    .reset_index()
)
loc_counts["total"] = loc_counts.groupby("search_location_id")["n"].transform("sum")
loc_counts["prob"] = loc_counts["n"] / loc_counts["total"]
loc_counts = loc_counts.sort_values(["search_location_id", "n"], ascending=[True, False])
# Keep top five transitions: they cover 96%+ of held-out clicked locations.
transition_probs = {
    int(source): {int(target): float(prob) for target, prob in zip(group.item_location_id, group.prob)}
    for source, group in loc_counts.groupby("search_location_id").head(5).groupby("search_location_id")
}

micro_counts = (
    profile.groupby(["normalized_query", "item_microcat_id"])
    .size()
    .rename("n")
    .reset_index()
)
micro_counts["total"] = micro_counts.groupby("normalized_query")["n"].transform("sum")
micro_counts["prob"] = micro_counts["n"] / micro_counts["total"]
micro_counts = micro_counts.sort_values(["normalized_query", "n"], ascending=[True, False])
exact_micro_probs = {
    query: {int(cat): float(prob) for cat, prob in zip(group.item_microcat_id, group.prob)}
    for query, group in micro_counts.groupby("normalized_query").head(5).groupby("normalized_query")
}

# A query-to-microcategory channel for queries without an exact history match.
# Each microcategory is represented by its most frequent historical queries.
query_cat_counts = (
    profile.groupby(["item_microcat_id", "normalized_query"])
    .size()
    .rename("n")
    .reset_index()
    .sort_values(["item_microcat_id", "n"], ascending=[True, False])
)
micro_docs = (
    query_cat_counts.groupby("item_microcat_id")
    .head(250)
    .groupby("item_microcat_id")["normalized_query"]
    .agg(" ".join)
)
micro_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80_000,
    sublinear_tf=True, dtype=np.float32,
)
micro_mat = micro_vec.fit_transform(micro_docs)
micro_qmat = micro_vec.transform(val_main)
micro_sim = (micro_qmat @ micro_mat.T).tocsr()
centroid_by_query = []
microcat_values = micro_docs.index.to_numpy()
for row in range(micro_sim.shape[0]):
    left, right = micro_sim.indptr[row], micro_sim.indptr[row + 1]
    vals, cols = micro_sim.data[left:right], micro_sim.indices[left:right]
    if len(vals) > 5:
        keep = np.argpartition(vals, -5)[-5:]
        vals, cols = vals[keep], cols[keep]
    centroid_by_query.append({int(microcat_values[col]): float(value) for col, value in zip(cols, vals)})
del micro_qmat, micro_mat, micro_vec, micro_sim

print("Preparing item texts", flush=True)
item_title = clean(items["item_title_raw"])
item_params = clean(items["item_infm_params_text"])
item_desc = clean(items["item_description_raw"]).str.slice(0, 1000)
item_mixed = item_title + " " + item_title + " " + item_params + " " + item_desc
# Raw text is no longer needed and would otherwise remain alongside every
# sparse matrix below.
items.drop(columns=["item_title_raw", "item_infm_params_text", "item_description_raw"], inplace=True)

# Fit a shared word vocabulary, but materialise only one field matrix at a
# time.  Shared IDF keeps the channels comparable without a multi-GB peak.
print("Fitting shared word index", flush=True)
word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
    max_features=200_000, sublinear_tf=True, dtype=np.float32,
)
word_vec.fit(item_mixed)

channels = {}
channel_scores = {}
for name, query_text, item_text in [
    ("mixed", val_mixed_text, item_mixed),
    ("title", val_main, item_title),
    ("params", val_main + " " + val_params + " " + val_params, item_params),
    ("description", val_main, item_desc),
]:
    print(f"Retrieving {name}", flush=True)
    qm = word_vec.transform(query_text)
    im = word_vec.transform(item_text)
    channels[name], channel_scores[name] = sparse_topk(qm, im)
    del qm, im
    gc.collect()
    if name == "mixed":
        del item_mixed
        gc.collect()

del word_vec
gc.collect()

print("Fitting character index", flush=True)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.995,
    max_features=180_000, sublinear_tf=True, dtype=np.float32,
)
char_item = char_vec.fit_transform(item_title + " " + item_params.str.slice(0, 300))
char_query = char_vec.transform(val_main)
channels["char"], channel_scores["char"] = sparse_topk(char_query, char_item)
del char_item, char_query, char_vec
gc.collect()

# Neural retrieval was computed on Kaggle GPU over exactly the same corpus and
# validation sample.  Fail loudly on any ordering mismatch: indices in the NPZ
# are row positions of benchmark_items.parquet, not item IDs.
print("Loading ML-Embed dense candidates", flush=True)
assert DENSE.exists(), f"Missing {DENSE}; run kaggle_ml_embed_2026.ipynb first"
dense = np.load(DENSE, allow_pickle=True)  # trusted artifact produced by our Kaggle notebook
assert np.array_equal(dense["item_ids"].astype(str), item_ids)
assert np.array_equal(dense["val_group_ids"].astype(str), validation.index.astype(str).to_numpy())
assert dense["val_indices"].shape[0] == len(validation)
channels["dense"] = [row.astype(np.int32, copy=False) for row in dense["val_indices"]]
channel_scores["dense"] = [row.astype(np.float32) for row in dense["val_scores"]]

reviews = items["item_rating_reviews_count"].fillna(0).to_numpy(dtype=float)
ratings = items["item_rating"].fillna(0).to_numpy(dtype=float)
popularity = np.log1p(reviews) * np.clip(ratings / 5, 0, 1)
popularity /= max(popularity.max(), 1)
review_norm = np.log1p(reviews)
review_norm /= max(review_norm.max(), 1)

records = []


def evaluate(name, builder, positions, phase):
    values = []
    hits = 0
    totals = 0
    for pos in positions:
        ranking = builder(pos, validation.iloc[pos])
        relevant = validation.iloc[pos]["relevant_idx"]
        values.append(recall(ranking, relevant))
        hits += len(set(ranking[:TOP_K]) & relevant)
        totals += len(relevant)
    macro = float(np.mean(values))
    micro = hits / totals
    records.append({"name": name, "phase": phase, "macro_recall50": macro, "micro_recall50": micro})
    print(f"{phase:5s} {name:60s} macro={macro:.6f} micro={micro:.6f}", flush=True)
    return macro


# Individual fields and candidate-union ceiling.
for name in channels:
    evaluate(name, lambda p, r, n=name: channels[n][p], test_positions, "test")

lexical_names = ["mixed", "title", "params", "description", "char"]
all_names = lexical_names + ["dense"]
# Oracle diagnostic over the complete union (up to 5 * 180 candidates), not an
# arbitrary first 50 from an unordered set.
union_values = []
union_hits = union_total = 0
for pos in test_positions:
    union = set().union(*(set(channels[n][pos]) for n in all_names))
    relevant = validation.iloc[pos]["relevant_idx"]
    hits = len(union & relevant)
    union_values.append(hits / len(relevant))
    union_hits += hits
    union_total += len(relevant)
records.append({
    "name": "union@all-channels oracle",
    "phase": "test",
    "macro_recall50": float(np.mean(union_values)),
    "micro_recall50": union_hits / union_total,
})
print(
    f"test  {'union@all-channels oracle':60s} "
    f"macro={np.mean(union_values):.6f} micro={union_hits / union_total:.6f}",
    flush=True,
)

# Tune a compact set of fielded RRF weights on tune; report chosen variants on
# untouched test groups.  We avoid an exhaustive weight search.
weight_configs = {
    "mixed+char": [1.0, 0.8],
    "title+params+desc+char": [1.2, 0.55, 0.55, 0.8],
    "fielded-title-heavy": [1.5, 0.45, 0.35, 0.9],
    "all-five": [0.65, 1.0, 0.45, 0.35, 0.75],
    "dense+mixed+char": [1.0, 0.8, 0.8],
    "dense+all-lexical": [1.0, 0.55, 0.8, 0.35, 0.25, 0.65],
}
config_channels = {
    "mixed+char": ["mixed", "char"],
    "title+params+desc+char": ["title", "params", "description", "char"],
    "fielded-title-heavy": ["title", "params", "description", "char"],
    "all-five": lexical_names,
    "dense+mixed+char": ["dense", "mixed", "char"],
    "dense+all-lexical": ["dense", "mixed", "title", "params", "description", "char"],
}


def make_builder(config, structured=None):
    names = config_channels[config]
    weights = weight_configs[config]
    structured = structured or {}

    def builder(pos, row):
        scores = rrf_score([channels[name][pos] for name in names], weights)
        add_structured(
            scores,
            row,
            val_main.iloc[pos],
            item_locs,
            item_microcats,
            transition_probs,
            exact_micro_probs,
            centroid_by_query[pos],
            popularity,
            **structured,
        )
        return rank_from_scores(scores)

    return builder


for config in weight_configs:
    evaluate(config, make_builder(config), tune_positions, "tune")
    evaluate(config, make_builder(config), test_positions, "test")

best_config = max(
    (record for record in records if record["phase"] == "tune" and record["name"] in weight_configs),
    key=lambda record: record["macro_recall50"],
)["name"]
print("Best text config:", best_config, flush=True)

# Structured reranking ablations. RRF values are roughly 0.005--0.03, hence
# bonuses in the 1e-3 range are meaningful but remain soft.
structured_configs = {
    "exact-location": {"exact_loc_bonus": 0.004},
    "transition-location": {"transition_bonus": 0.006},
    "transition+exact": {"exact_loc_bonus": 0.0015, "transition_bonus": 0.005},
    "exact-microcat": {"exact_micro_bonus": 0.005},
    "centroid-microcat": {"centroid_micro_bonus": 0.003},
    "microcats-both": {"exact_micro_bonus": 0.005, "centroid_micro_bonus": 0.002},
    "location+microcats": {
        "exact_loc_bonus": 0.0015,
        "transition_bonus": 0.005,
        "exact_micro_bonus": 0.005,
        "centroid_micro_bonus": 0.002,
    },
    "location+microcats+pop": {
        "exact_loc_bonus": 0.0015,
        "transition_bonus": 0.005,
        "exact_micro_bonus": 0.005,
        "centroid_micro_bonus": 0.002,
        "popularity_bonus": 0.0005,
    },
}
for suffix, kwargs in structured_configs.items():
    name = f"{best_config} + {suffix}"
    evaluate(name, make_builder(best_config, kwargs), tune_positions, "tune")

best_structured_name = max(
    (
        record
        for record in records
        if record["phase"] == "tune" and record["name"].startswith(best_config + " +")
    ),
    key=lambda record: record["macro_recall50"],
)["name"]
best_suffix = best_structured_name.split(" + ", 1)[1]
best_kwargs = structured_configs[best_suffix]
evaluate(best_structured_name, make_builder(best_config, best_kwargs), test_positions, "test")

# Quota ablation on top of a larger RRF pool (top 180 before quota).
def quota_builder(quota):
    base = make_builder(best_config, best_kwargs)

    def builder(pos, row):
        # Recreate the same scores but retain a deeper list.
        names = config_channels[best_config]
        scores = rrf_score([channels[name][pos] for name in names], weight_configs[best_config])
        add_structured(
            scores, row, val_main.iloc[pos], item_locs, item_microcats,
            transition_probs, exact_micro_probs, centroid_by_query[pos], popularity,
            **best_kwargs,
        )
        deep = rank_from_scores(scores, limit=RETRIEVAL_K)
        return quota_ranking(deep, item_locs, int(row["search_location_id"]), quota)

    return builder


for quota in (20, 30, 40):
    evaluate(f"best + exact-location quota {quota}", quota_builder(quota), test_positions, "test")

# Checkpoint the expensive retrieval results before the optional supervised
# tail.  A later interruption must not erase several minutes of experiments.
OUT.write_text(
    json.dumps(
        {
            "status": "structured_checkpoint",
            "settings": {
                "seed": SEED,
                "groups": len(validation),
                "tune_groups": len(tune_positions),
                "test_groups": len(test_positions),
                "retrieval_k": RETRIEVAL_K,
                "top_k": TOP_K,
                "best_text_config": best_config,
                "best_structured_config": best_structured_name,
            },
            "results": records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Lightweight pointwise reranker. It learns only on tune groups and is assessed
# on untouched test groups. Candidate features are ranks/scores from each text
# channel plus the leakage-safe structural signals.
def candidate_features(pos, row):
    candidate_set = set()
    rank_maps, score_maps = {}, {}
    for name in all_names:
        ranking = channels[name][pos]
        values = channel_scores[name][pos]
        candidate_set.update(map(int, ranking))
        rank_maps[name] = {int(idx): rank for rank, idx in enumerate(ranking)}
        score_maps[name] = {int(idx): float(value) for idx, value in zip(ranking, values)}

    source_loc = int(row["search_location_id"])
    trans = transition_probs.get(source_loc, {})
    exact_micro = exact_micro_probs.get(val_main.iloc[pos], {})
    centroid_micro = centroid_by_query[pos]
    candidates = np.fromiter(candidate_set, dtype=np.int32)
    features = []
    for idx in candidates:
        values = []
        for name in all_names:
            rank = rank_maps[name].get(int(idx), RETRIEVAL_K + 50)
            values.extend([1.0 / (rank + 1), score_maps[name].get(int(idx), 0.0)])
        values.extend(
            [
                float(item_locs[idx] == source_loc),
                trans.get(int(item_locs[idx]), 0.0),
                exact_micro.get(int(item_microcats[idx]), 0.0),
                centroid_micro.get(int(item_microcats[idx]), 0.0),
                popularity[idx],
                review_norm[idx],
                ratings[idx] / 5,
            ]
        )
        features.append(values)
    return candidates, np.asarray(features, dtype=np.float32)


print("Training streaming supervised reranker", flush=True)
reranker = SGDClassifier(
    loss="log_loss", penalty="l2", alpha=3e-5, random_state=SEED,
    average=True,
)
first_batch = True
for epoch in range(2):
    for start in range(0, len(tune_positions), 64):
        batch_x, batch_y = [], []
        for pos in tune_positions[start : start + 64]:
            candidates, feats = candidate_features(pos, validation.iloc[pos])
            labels = np.fromiter(
                (idx in validation.iloc[pos]["relevant_idx"] for idx in candidates),
                dtype=np.int8,
            )
            batch_x.append(feats)
            batch_y.append(labels)
        batch_x = np.vstack(batch_x)
        batch_y = np.concatenate(batch_y)
        # Roughly one positive per several hundred hard candidates.
        sample_weight = np.where(batch_y == 1, 120.0, 1.0)
        kwargs = {"classes": np.array([0, 1])} if first_batch else {}
        reranker.partial_fit(batch_x, batch_y, sample_weight=sample_weight, **kwargs)
        first_batch = False
        del batch_x, batch_y, sample_weight


def supervised_ranking(pos, row, limit=TOP_K):
    candidates, feats = candidate_features(pos, row)
    pred = reranker.predict_proba(feats)[:, 1]
    return candidates[np.argsort(pred)[::-1][:limit]].tolist()


def supervised_builder(pos, row):
    return supervised_ranking(pos, row, TOP_K)


evaluate("logistic supervised reranker", supervised_builder, test_positions, "test")

# The hard location quota was useful for the hand-tuned fusion, but the model
# already sees location features.  Test explicitly instead of assuming that the
# same post-processing still helps.
for quota in (20, 30, 40):
    evaluate(
        f"logistic reranker + exact-location quota {quota}",
        lambda p, r, q=quota: quota_ranking(
            supervised_ranking(p, r, RETRIEVAL_K),
            item_locs,
            int(r["search_location_id"]),
            q,
        ),
        test_positions,
        "test",
    )

# A realistic history diagnostic: build the history channel only from tune
# labels and evaluate transfer to different held-out contexts.  This avoids
# giving a test row its own clicked item while mimicking the information that
# is available for the real benchmark.
history_counts = defaultdict(lambda: defaultdict(int))
for pos in tune_positions:
    query = val_main.iloc[pos]
    for idx in validation.iloc[pos]["relevant_idx"]:
        history_counts[query][int(idx)] += 1
history_rankings = {
    query: [idx for idx, _ in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)]
    for query, counts in history_counts.items()
}


def history_fusion_builder(history_weight, quota=None):
    def builder(pos, row):
        learned = supervised_ranking(pos, row, RETRIEVAL_K)
        historical = history_rankings.get(val_main.iloc[pos], [])
        scores = rrf_score([learned, historical], [1.0, history_weight])
        deep = rank_from_scores(scores, RETRIEVAL_K)
        if quota is not None:
            deep = quota_ranking(deep, item_locs, int(row["search_location_id"]), quota)
        return deep[:TOP_K]

    return builder


for weight in (0.5, 1.0, 2.0):
    evaluate(
        f"logistic + leakage-safe history RRF weight {weight}",
        history_fusion_builder(weight),
        test_positions,
        "test",
    )

# Diagnostics by query-history and exact-location availability.
item_location_set = set(map(int, item_locs))
best_builder = make_builder(best_config, best_kwargs)
segments = {
    "has_exact_query_history": [p for p in test_positions if val_main.iloc[p] in exact_micro_probs],
    "no_exact_query_history": [p for p in test_positions if val_main.iloc[p] not in exact_micro_probs],
    "exact_location_exists": [p for p in test_positions if int(validation.iloc[p]["search_location_id"]) in item_location_set],
    "exact_location_absent": [p for p in test_positions if int(validation.iloc[p]["search_location_id"]) not in item_location_set],
}
for segment, positions in segments.items():
    if positions:
        evaluate(f"best segment: {segment} (n={len(positions)})", best_builder, positions, "segment")

output = {
    "settings": {
        "seed": SEED,
        "groups": len(validation),
        "tune_groups": len(tune_positions),
        "test_groups": len(test_positions),
        "retrieval_k": RETRIEVAL_K,
        "top_k": TOP_K,
        "best_text_config": best_config,
        "best_structured_config": best_structured_name,
        "runtime_seconds": time.time() - started,
    },
    "coverage": {
        "validation_exact_query_profile": float(val_main.isin(exact_micro_probs).mean()),
        "validation_exact_location_in_items": float(validation["search_location_id"].isin(item_location_set).mean()),
    },
    "reranker": {
        "feature_order": [
            *[part for name in all_names for part in (f"{name}_reciprocal_rank", f"{name}_cosine")],
            "exact_location",
            "location_transition_probability",
            "exact_query_microcat_probability",
            "microcat_centroid_similarity",
            "popularity",
            "log_reviews",
            "rating",
        ],
        "coef": reranker.coef_[0].astype(float).tolist(),
        "intercept": reranker.intercept_.astype(float).tolist(),
    },
    "results": records,
}
OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}; runtime={time.time() - started:.1f}s", flush=True)
