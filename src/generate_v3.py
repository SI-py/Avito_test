"""Generate the v3 Avito candidate-retrieval submission.

The script reuses the leakage-safe linear reranker fitted in
``hybrid_experiments.py``.  It retrieves a broad union of five lexical
channels, scores candidates with learned structural features, softly fuses
historical clicks, and applies a small exact-location quota.

Peak memory is kept below the earlier implementation by materialising one
sparse item field at a time.
"""

from __future__ import annotations

from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA = Path("data")
MODEL = Path("hybrid_results.json")
OUTPUT = Path("answer_v3.csv")
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
    """Return item positions and cosine scores for each sparse query row."""
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


def rrf(rankings, weights, limit=RETRIEVAL_K, rrf_k=60):
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[int(idx)] += weight / (rrf_k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]]


def location_quota(ranking, item_locations, query_location, quota=20):
    """Reserve up to ``quota`` places for exact-location candidates."""
    local = [idx for idx in ranking if item_locations[idx] == query_location]
    chosen = local[:quota]
    used = set(chosen)
    for idx in ranking:
        if idx not in used:
            chosen.append(idx)
            used.add(idx)
        if len(chosen) == TOP_K:
            break
    return chosen


print("Loading data and learned coefficients", flush=True)
model = json.loads(MODEL.read_text(encoding="utf-8"))["reranker"]
coef = np.asarray(model["coef"], dtype=np.float32)
intercept = float(model["intercept"][0])

train_cols = [
    "search_query", "search_location_id", "item_id", "item_location_id",
    "item_microcat_id",
]
train = pd.read_parquet(DATA / "train.parquet", columns=train_cols)
queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
items = pd.read_parquet(DATA / "benchmark_items.parquet")

item_ids = items["item_id"].astype(str).to_numpy()
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
benchmark_item_ids = set(item_ids)
item_locations = items["item_location_id"].to_numpy()
item_microcats = items["item_microcat_id"].to_numpy()

query_main = clean(queries["search_query"])
query_params = clean(queries["search_infm_params_text"])
query_mixed = query_main + " " + query_main + " " + query_params

print("Building leakage-safe behaviour profiles", flush=True)
profile = train.loc[
    ~train["item_id"].isin(benchmark_item_ids),
    ["search_query", "search_location_id", "item_location_id", "item_microcat_id"],
].copy()
profile["normalized_query"] = clean(profile["search_query"])

loc_counts = (
    profile.groupby(["search_location_id", "item_location_id"])
    .size().rename("n").reset_index()
)
loc_counts["total"] = loc_counts.groupby("search_location_id")["n"].transform("sum")
loc_counts["prob"] = loc_counts["n"] / loc_counts["total"]
loc_counts = loc_counts.sort_values(["search_location_id", "n"], ascending=[True, False])
transition_probs = {
    int(source): {int(target): float(prob) for target, prob in zip(group.item_location_id, group.prob)}
    for source, group in loc_counts.groupby("search_location_id").head(5).groupby("search_location_id")
}

micro_counts = (
    profile.groupby(["normalized_query", "item_microcat_id"])
    .size().rename("n").reset_index()
)
micro_counts["total"] = micro_counts.groupby("normalized_query")["n"].transform("sum")
micro_counts["prob"] = micro_counts["n"] / micro_counts["total"]
micro_counts = micro_counts.sort_values(["normalized_query", "n"], ascending=[True, False])
exact_micro_probs = {
    query: {int(cat): float(prob) for cat, prob in zip(group.item_microcat_id, group.prob)}
    for query, group in micro_counts.groupby("normalized_query").head(5).groupby("normalized_query")
}

# Query-to-microcategory centroids cover unseen queries.
query_cat_counts = (
    profile.groupby(["item_microcat_id", "normalized_query"])
    .size().rename("n").reset_index()
    .sort_values(["item_microcat_id", "n"], ascending=[True, False])
)
micro_docs = (
    query_cat_counts.groupby("item_microcat_id").head(250)
    .groupby("item_microcat_id")["normalized_query"].agg(" ".join)
)
micro_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2,
    max_features=80_000, sublinear_tf=True, dtype=np.float32,
)
micro_mat = micro_vec.fit_transform(micro_docs)
micro_similarity = (micro_vec.transform(query_main) @ micro_mat.T).tocsr()
microcat_values = micro_docs.index.to_numpy()
centroid_by_query = []
for row in range(micro_similarity.shape[0]):
    left, right = micro_similarity.indptr[row], micro_similarity.indptr[row + 1]
    values = micro_similarity.data[left:right]
    columns = micro_similarity.indices[left:right]
    if len(values) > 5:
        keep = np.argpartition(values, -5)[-5:]
        values, columns = values[keep], columns[keep]
    centroid_by_query.append(
        {int(microcat_values[col]): float(value) for col, value in zip(columns, values)}
    )
del micro_vec, micro_mat, micro_similarity, query_cat_counts, micro_docs

# Historical benchmark items are a separate, softly fused channel.  They are
# supplied training evidence, not reconstructed test labels.
history = train.loc[train["item_id"].isin(benchmark_item_ids), ["search_query", "item_id"]].copy()
history["normalized_query"] = clean(history["search_query"])
history_counts = (
    history.groupby(["normalized_query", "item_id"]).size().rename("n").reset_index()
    .sort_values(["normalized_query", "n"], ascending=[True, False])
)
history_lookup = {
    query: [item_id_to_idx[item_id] for item_id in group["item_id"]]
    for query, group in history_counts.groupby("normalized_query", sort=False)
}
del history, history_counts, train
gc.collect()

print("Preparing item text", flush=True)
item_title = clean(items["item_title_raw"])
item_params = clean(items["item_infm_params_text"])
item_description = clean(items["item_description_raw"]).str.slice(0, 1000)
item_mixed = item_title + " " + item_title + " " + item_params + " " + item_description
items.drop(columns=["item_title_raw", "item_infm_params_text", "item_description_raw"], inplace=True)

channels = {}
channel_scores = {}
print("Fitting word index", flush=True)
word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
    max_features=200_000, sublinear_tf=True, dtype=np.float32,
)
word_vec.fit(item_mixed)
for name, query_text, item_text in [
    ("mixed", query_mixed, item_mixed),
    ("title", query_main, item_title),
    ("params", query_main + " " + query_params + " " + query_params, item_params),
    ("description", query_main, item_description),
]:
    print(f"Retrieving {name}", flush=True)
    query_matrix = word_vec.transform(query_text)
    item_matrix = word_vec.transform(item_text)
    channels[name], channel_scores[name] = sparse_topk(query_matrix, item_matrix)
    del query_matrix, item_matrix
    gc.collect()
    if name == "mixed":
        del item_mixed
del word_vec
gc.collect()

print("Retrieving char n-grams", flush=True)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.995,
    max_features=180_000, sublinear_tf=True, dtype=np.float32,
)
char_items = char_vec.fit_transform(item_title + " " + item_params.str.slice(0, 300))
char_queries = char_vec.transform(query_main)
channels["char"], channel_scores["char"] = sparse_topk(char_queries, char_items)
del char_vec, char_items, char_queries, item_title, item_params, item_description
gc.collect()

reviews = items["item_rating_reviews_count"].fillna(0).to_numpy(dtype=float)
ratings = items["item_rating"].fillna(0).to_numpy(dtype=float)
popularity = np.log1p(reviews) * np.clip(ratings / 5, 0, 1)
popularity /= max(popularity.max(), 1)
review_norm = np.log1p(reviews)
review_norm /= max(review_norm.max(), 1)
fallback = np.argsort(popularity)[::-1].tolist()

all_names = ["mixed", "title", "params", "description", "char"]
expected_features = [
    *[part for name in all_names for part in (f"{name}_reciprocal_rank", f"{name}_cosine")],
    "exact_location", "location_transition_probability",
    "exact_query_microcat_probability", "microcat_centroid_similarity",
    "popularity", "log_reviews", "rating",
]
assert model["feature_order"] == expected_features
assert len(coef) == len(expected_features)

print("Scoring and fusing candidates", flush=True)
final_rankings = []
for pos, row in queries.iterrows():
    candidate_set = set()
    rank_maps, score_maps = {}, {}
    for name in all_names:
        ranking = channels[name][pos]
        values = channel_scores[name][pos]
        candidate_set.update(map(int, ranking))
        rank_maps[name] = {int(idx): rank for rank, idx in enumerate(ranking)}
        score_maps[name] = {int(idx): float(value) for idx, value in zip(ranking, values)}

    source_location = int(row["search_location_id"])
    transitions = transition_probs.get(source_location, {})
    normalized_query = query_main.iloc[pos]
    exact_micro = exact_micro_probs.get(normalized_query, {})
    centroid_micro = centroid_by_query[pos]
    candidates = np.fromiter(candidate_set, dtype=np.int32)
    features = []
    for idx in candidates:
        values = []
        for name in all_names:
            rank = rank_maps[name].get(int(idx), RETRIEVAL_K + 50)
            values.extend([1.0 / (rank + 1), score_maps[name].get(int(idx), 0.0)])
        values.extend([
            float(item_locations[idx] == source_location),
            transitions.get(int(item_locations[idx]), 0.0),
            exact_micro.get(int(item_microcats[idx]), 0.0),
            centroid_micro.get(int(item_microcats[idx]), 0.0),
            popularity[idx], review_norm[idx], ratings[idx] / 5,
        ])
        features.append(values)
    features = np.asarray(features, dtype=np.float32)
    learned_scores = features @ coef + intercept
    learned = candidates[np.argsort(learned_scores)[::-1][:RETRIEVAL_K]].tolist()

    # Weight 1.0 was the best conservative setting on leakage-safe holdout.
    historical = history_lookup.get(normalized_query, [])
    fused = rrf([learned, historical], [1.0, 1.0], RETRIEVAL_K)
    chosen = location_quota(fused, item_locations, source_location, quota=20)

    seen = set(chosen)
    if len(chosen) < TOP_K:
        for idx in fallback:
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
            if len(chosen) == TOP_K:
                break
    final_rankings.append(chosen[:TOP_K])

answer = pd.DataFrame({
    "query_id": queries["query_id"].astype(str),
    "answer": [" ".join(item_ids[ranking]) for ranking in final_rankings],
})
answer.to_csv(OUTPUT, index=False, encoding="utf-8")

# Strict submission checks.  IDs must remain strings to preserve leading zeros.
saved = pd.read_csv(OUTPUT, dtype=str, keep_default_na=False)
assert saved.columns.tolist() == ["query_id", "answer"]
assert len(saved) == len(queries)
assert saved["query_id"].tolist() == queries["query_id"].astype(str).tolist()
assert saved["query_id"].is_unique
for query_id, value in saved.itertuples(index=False):
    ids = value.split(" ")
    assert len(ids) == TOP_K
    assert len(ids) == len(set(ids))
    assert set(ids) <= benchmark_item_ids
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in ids), query_id

old = pd.read_csv("answer.csv", dtype=str, keep_default_na=False)
overlap = np.mean([
    len(set(left.split()) & set(right.split()))
    for left, right in zip(old["answer"], saved["answer"])
])
print(f"{OUTPUT} is valid: {len(saved)} rows, exactly {TOP_K} IDs per row", flush=True)
print(f"Mean overlap with leaderboard v1: {overlap:.2f}/{TOP_K}", flush=True)
