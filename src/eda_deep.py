"""Deep, reproducible EDA for the Avito candidate-generation task.

The script is deliberately analysis-only: it reads the three parquet files and
writes a Markdown report.  It never touches solution.ipynb or answer*.csv.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORT = ROOT / "eda_report.md"


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def quantiles(s: pd.Series, qs=(0, .25, .5, .75, .9, .95, .99, 1)) -> dict:
    return {str(q): float(v) for q, v in s.quantile(qs).items()}


def md_table(rows, columns=None) -> str:
    df = pd.DataFrame(rows, columns=columns)
    return df.to_markdown(index=False)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def main():
    print("Loading parquet files")
    train_cols = [
        "search_query", "search_location_id", "search_is_delivery_search",
        "search_infm_params_text", "search_category", "item_title_raw",
        "item_microcat_id", "item_location_id",
        "item_id", "item_category_id",
    ]
    item_cols = [
        "item_title_raw", "item_microcat_id", "item_location_id",
        "item_infm_params_text", "item_id", "item_category_id",
    ]
    train = pd.read_parquet(DATA / "train.parquet", columns=train_cols)
    queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=item_cols)

    # Normalized fields are intentionally simple and exactly reproducible.
    for df in (train, queries):
        df["q_norm"] = df["search_query"].map(norm)
        df["qp_norm"] = df["search_infm_params_text"].map(norm)
        df["q_tokens"] = df["q_norm"].str.split().map(len)
        df["q_chars"] = df["q_norm"].str.len()
    # Avoid materialising normalized copies of large train item text columns.
    # Current-corpus fields are enough for lexical analysis of surviving positives.
    items["title_norm"] = items["item_title_raw"].map(norm)
    items["params_norm"] = items["item_infm_params_text"].map(norm)

    key_cols = ["q_norm", "search_location_id", "search_is_delivery_search", "qp_norm", "search_category"]
    raw_key_cols = ["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category"]

    train_full_key_rows = train[key_cols].drop_duplicates()
    train_full_keys = pd.MultiIndex.from_frame(train_full_key_rows)
    bench_full_keys = pd.MultiIndex.from_frame(queries[key_cols])
    train_raw_keys = pd.MultiIndex.from_frame(train[raw_key_cols].fillna("").drop_duplicates())
    bench_raw_keys = pd.MultiIndex.from_frame(queries[raw_key_cols].fillna(""))

    train_q = set(train.q_norm)
    train_raw_q = set(train.search_query.fillna(""))
    benchmark_ids = set(items.item_id.astype(str))
    train_ids = set(train.item_id.astype(str))

    # Group statistics / ambiguity.
    by_q = train.groupby("q_norm", sort=False).agg(
        rows=("item_id", "size"),
        items=("item_id", "nunique"),
        microcats=("item_microcat_id", "nunique"),
        locations=("search_location_id", "nunique"),
        titles=("item_title_raw", "nunique"),
    )
    by_full = train.groupby(key_cols, dropna=False, sort=False).agg(
        rows=("item_id", "size"), items=("item_id", "nunique"), microcats=("item_microcat_id", "nunique")
    )
    mc_counts = train.groupby(["q_norm", "item_microcat_id"], sort=False).size().rename("n")
    mc_total = mc_counts.groupby(level=0).sum()
    mc_max = mc_counts.groupby(level=0).max()
    dominant_mc_share = mc_max / mc_total
    probs = mc_counts / mc_counts.groupby(level=0).transform("sum")
    mc_entropy = (-(probs * np.log2(probs))).groupby(level=0).sum()

    # Historical exact-query channel and its oracle upper bounds.
    hist_bench = train[train.item_id.astype(str).isin(benchmark_ids)].copy()
    hist_by_q = hist_bench.groupby("q_norm").item_id.agg(lambda x: set(map(str, x)))
    hist_by_full = hist_bench.groupby(key_cols, dropna=False).item_id.agg(lambda x: set(map(str, x)))
    bench_q_hist_counts = queries.q_norm.map(hist_by_q.map(len)).fillna(0).astype(int)
    # Note: for a known full query, all historical current-corpus positives fit in 50 surprisingly often.
    full_hist_len = hist_by_full.map(len)

    # Click popularity and head concentration.
    click_counts_all = train.item_id.astype(str).value_counts()
    click_counts_bench = hist_bench.item_id.astype(str).value_counts()
    sorted_clicks = click_counts_all.sort_values(ascending=False)
    cum = sorted_clicks.cumsum() / sorted_clicks.sum()
    head_share = {}
    for frac in (.001, .01, .05, .10, .20):
        n = max(1, math.ceil(len(sorted_clicks) * frac))
        head_share[frac] = float(sorted_clicks.iloc[:n].sum() / sorted_clicks.sum())

    # Location / remote behavior on observed positive pairs.
    same_loc = train.search_location_id.eq(train.item_location_id)
    remote_re = r"удален|дистанц|онлайн|remote|по всей россии|вся россия"
    corpus_remote = (
        items.params_norm.str.contains(remote_re, regex=True)
        | items.title_norm.str.contains(remote_re, regex=True)
    )
    remote_ids = set(items.loc[corpus_remote, "item_id"])
    remote_item = train.item_id.isin(remote_ids)
    same_loc_by_mc = (
        train.assign(same=same_loc)
        .groupby("item_microcat_id")
        .agg(rows=("same", "size"), same_rate=("same", "mean"))
        .query("rows >= 100")
        .sort_values("same_rate")
    )
    # Distance only for rows with usable coordinates and mismatching IDs.
    # Search coordinates are not supplied, so geographic distance cannot be computed.

    # Query features and data shift.
    train_unique_q = train.drop_duplicates(key_cols)
    shift_rows = []
    for label, df in [("train unique full queries", train_unique_q), ("benchmark", queries)]:
        shift_rows.append({
            "split": label,
            "rows": len(df),
            "unique normalized text": df.q_norm.nunique(),
            "params nonempty": pct(df.qp_norm.ne("").mean()),
            "delivery=1": pct(df.search_is_delivery_search.eq(1).mean()),
            "median tokens": float(df.q_tokens.median()),
            "p90 tokens": float(df.q_tokens.quantile(.9)),
            "unique locations": df.search_location_id.nunique(),
        })

    # Lexical coverage. For positive train pairs in current corpus, measure whether
    # all/any query tokens occur in each field. Stopwords are retained because the
    # production vectorizer also currently retains them.
    # Description coverage is expensive; 50k deterministic positive pairs give
    # a tight estimate without duplicating hundreds of MB of text in memory.
    val = hist_bench[["q_norm", "qp_norm", "item_id", "item_microcat_id", "search_location_id", "item_location_id"]]
    if len(val) > 50_000:
        val = val.sample(50_000, random_state=42)
    else:
        val = val.copy()
    # Load the corpus description column once, only after the large train table
    # has been reduced to the columns used by the analysis.
    item_desc = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id", "item_description_raw"])
    val = val.merge(
        items[["item_id", "title_norm", "params_norm"]], on="item_id", how="left", validate="many_to_one"
    ).merge(item_desc, on="item_id", how="left", validate="many_to_one")
    val["desc_norm"] = val.item_description_raw.map(norm)
    def coverage_row(row):
        toks = set(row.q_norm.split())
        if not toks:
            return (False,) * 8
        fields = [set(row.title_norm.split()), set(row.params_norm.split()), set(row.desc_norm.split())]
        union = fields[0] | fields[1] | fields[2]
        return (
            toks <= fields[0], bool(toks & fields[0]),
            toks <= fields[1], bool(toks & fields[1]),
            toks <= fields[2], bool(toks & fields[2]),
            toks <= union, bool(toks & union),
        )
    cov_cols = ["all_title", "any_title", "all_params", "any_params", "all_desc", "any_desc", "all_anyfield", "any_anyfield"]
    cov = pd.DataFrame([coverage_row(r) for r in val.itertuples(index=False)], columns=cov_cols)

    # Token OOV relative to benchmark corpus by field.
    q_token_counts = Counter(tok for q in queries.q_norm for tok in q.split())
    title_vocab = set(tok for s in items.title_norm for tok in s.split())
    params_vocab = set(tok for s in items.params_norm for tok in s.split())
    # We only need to know which *query* tokens appear in descriptions. Scanning
    # for this small vocabulary is much more memory-efficient than retaining the
    # complete description vocabulary.
    query_vocab = set(q_token_counts)
    desc_vocab = set()
    token_re = re.compile(r"[0-9a-zа-я]+")
    for raw in item_desc.item_description_raw.fillna(""):
        desc_vocab.update(set(token_re.findall(unicodedata.normalize("NFKC", raw).lower().replace("ё", "е"))) & query_vocab)
    all_vocab = title_vocab | params_vocab | desc_vocab
    total_q_token_occ = sum(q_token_counts.values())
    oov_stats = {
        "title": sum(n for t, n in q_token_counts.items() if t not in title_vocab) / total_q_token_occ,
        "params": sum(n for t, n in q_token_counts.items() if t not in params_vocab) / total_q_token_occ,
        "description": sum(n for t, n in q_token_counts.items() if t not in desc_vocab) / total_q_token_occ,
        "union": sum(n for t, n in q_token_counts.items() if t not in all_vocab) / total_q_token_occ,
    }
    bench_zero_union = queries.q_norm.map(lambda s: not bool(set(s.split()) & all_vocab)).mean()

    # Item duplication and attribute consistency.
    title_groups = items.groupby("title_norm").agg(
        items=("item_id", "nunique"),
        microcats=("item_microcat_id", "nunique"),
        locations=("item_location_id", "nunique"),
    )
    nonempty_title = title_groups.index.to_numpy() != ""
    duplicate_title_rows = int(title_groups.loc[nonempty_title & title_groups["items"].gt(1), "items"].sum())
    duplicate_title_groups = title_groups.loc[nonempty_title & title_groups["items"].gt(1)]
    # Hashing raw content avoids another full in-memory normalized-description copy.
    exact_content_hash = pd.util.hash_pandas_object(
        pd.DataFrame({
            "item_title_raw": items.item_title_raw,
            "item_infm_params_text": items.item_infm_params_text,
            "item_description_raw": item_desc.item_description_raw,
        }).fillna(""), index=False
    )
    exact_content_counts = exact_content_hash.value_counts()
    duplicated_content_rows = int(exact_content_counts[exact_content_counts.gt(1)].sum())

    id_consistency = train.groupby("item_id").agg(
        titles=("item_title_raw", "nunique"), microcats=("item_microcat_id", "nunique"),
        locations=("item_location_id", "nunique"), categories=("item_category_id", "nunique")
    )

    # Params: exact benchmark filters seen in train and crude value-match with positive items.
    param_nonempty = train.qp_norm.ne("")
    bench_param_nonempty = queries.qp_norm.ne("")
    train_param_set = set(train.loc[param_nonempty, "qp_norm"])
    bench_param_seen = queries.loc[bench_param_nonempty, "qp_norm"].isin(train_param_set)
    # String containment is intentionally strict; it quantifies only easy wins.
    param_contained = val.loc[val.qp_norm.ne("")].apply(lambda r: r.qp_norm in r.params_norm, axis=1)

    # Bench-query slices useful for gated models.
    known_q_mask = queries.q_norm.isin(train_q)
    known_full_mask = bench_full_keys.isin(train_full_keys)
    raw_full_mask = bench_raw_keys.isin(train_raw_keys)
    known_loc_mask = queries.search_location_id.isin(set(train.search_location_id))
    known_q_item_history = bench_q_hist_counts.gt(0)

    # Examples rather than anonymized aggregates alone.
    ambiguous_examples = (
        by_q.join(dominant_mc_share.rename("dominant_mc_share")).join(mc_entropy.rename("mc_entropy"))
        .query("rows >= 10 and microcats >= 3")
        .sort_values(["dominant_mc_share", "rows"], ascending=[True, False])
        .head(15)
        .reset_index()
    )
    broad_title_examples = duplicate_title_groups.sort_values("items", ascending=False).head(12).reset_index()
    low_loc_examples = same_loc_by_mc.head(15).reset_index()
    high_loc_examples = same_loc_by_mc.tail(15).sort_values("same_rate", ascending=False).reset_index()

    lines = []
    A = lines.append
    A("# Deep EDA: Avito Recall@50 candidate generation\n")
    A("Generated by `eda_deep.py`. Counts use a transparent normalization: Unicode NFKC, lowercase, `ё→е`, punctuation→spaces.\n")
    A("## 1. Dataset shape and invariants\n")
    A(md_table([
        ["train", len(train), train.item_id.nunique(), train.q_norm.nunique(), train.search_location_id.nunique(), train.item_microcat_id.nunique()],
        ["benchmark queries", len(queries), "—", queries.q_norm.nunique(), queries.search_location_id.nunique(), "—"],
        ["benchmark items", len(items), items.item_id.nunique(), "—", items.item_location_id.nunique(), items.item_microcat_id.nunique()],
    ], ["split", "rows", "unique item_id", "unique q_norm", "unique locations", "unique microcats"]))
    A("")
    A(f"- Search categories: train={train.search_category.nunique()}, benchmark={queries.search_category.nunique()}; item categories: train={train.item_category_id.nunique()}, corpus={items.item_category_id.nunique()}.")
    A(f"- Exact duplicate rows in train: {train.duplicated().sum():,}.")
    A(f"- Train unique full query contexts: {len(by_full):,}; unique normalized texts: {len(by_q):,}.")
    A(f"- Item IDs shared by train and current corpus: {len(train_ids & benchmark_ids):,} / {len(benchmark_ids):,} corpus items ({pct(len(train_ids & benchmark_ids)/len(benchmark_ids))}).")

    A("\n## 2. Benchmark overlap with historical queries\n")
    A(md_table([
        ["raw search text seen", pct(queries.search_query.fillna("").isin(train_raw_q).mean()), int(queries.search_query.fillna("").isin(train_raw_q).sum())],
        ["normalized search text seen", pct(known_q_mask.mean()), int(known_q_mask.sum())],
        ["raw full context seen", pct(raw_full_mask.mean()), int(raw_full_mask.sum())],
        ["normalized full context seen", pct(known_full_mask.mean()), int(known_full_mask.sum())],
        ["query has historical clicked item still in corpus", pct(known_q_item_history.mean()), int(known_q_item_history.sum())],
        ["search location seen in train", pct(known_loc_mask.mean()), int(known_loc_mask.sum())],
    ], ["condition", "share", "queries"]))
    A("")
    A(f"- Historical current-corpus candidates per benchmark query: {quantiles(bench_q_hist_counts)}.")
    A(f"- Among normalized full contexts with any surviving historical item, share having <=50 distinct historical candidates: {pct((full_hist_len <= 50).mean())}; max={full_hist_len.max()}.")
    A("- Exact history is therefore a high-precision channel, but query-text-only history can mix cities and filter contexts; use full context first, then relax in stages.")

    A("\n## 3. Query repetition and ambiguity\n")
    A(f"- Rows per normalized query: {quantiles(by_q.rows)}.")
    A(f"- Distinct clicked items per normalized query: {quantiles(by_q['items'])}.")
    A(f"- Distinct microcategories per normalized query: {quantiles(by_q.microcats)}.")
    A(f"- Dominant-microcategory click share: {quantiles(dominant_mc_share)}.")
    A(f"- Weighted by benchmark queries, exact-text train profiles have median dominant-microcat share {queries.q_norm.map(dominant_mc_share).median():.3f}.")
    A("\nAmbiguous frequent queries (low dominant microcategory share):\n")
    A(ambiguous_examples[["q_norm", "rows", "items", "microcats", "locations", "dominant_mc_share", "mc_entropy"]].to_markdown(index=False))

    A("\n## 4. Geography and remote services\n")
    A(f"- Positive train pairs with exact location ID match: **{pct(same_loc.mean())}**.")
    A(f"- Remote/online wording appears in {pct(remote_item.mean())} of positive rows.")
    A(f"- Location match when remote wording exists: {pct(same_loc[remote_item].mean())}; without it: {pct(same_loc[~remote_item].mean())}.")
    A(f"- Mismatched-location positives containing remote wording: {pct(remote_item[~same_loc].mean())}.")
    A("- Search coordinates are absent, so `item_latitude/longitude` cannot yield true query-item distance. Location ID is the directly usable signal.")
    A("\nMicrocategories with lowest location-match rate (>=100 positives):\n")
    A(low_loc_examples.to_markdown(index=False))
    A("\nMicrocategories with highest location-match rate (>=100 positives):\n")
    A(high_loc_examples.to_markdown(index=False))
    A("- A single global location bonus is structurally suboptimal: use a microcategory- or query-conditioned location prior, and reduce it for remote/online intent.")

    A("\n## 5. Query/filter shift\n")
    A(md_table(shift_rows))
    A("")
    A(f"- Nonempty benchmark filters whose exact normalized string was seen in train: {pct(bench_param_seen.mean())} ({bench_param_seen.sum():,}/{len(bench_param_seen):,}).")
    A(f"- On positive train rows with filters, the whole normalized query-filter string is literally contained in item params only {pct(param_contained.mean())}. Exact containment is too brittle.")
    A("- Params need parsing or token/value retrieval, not whole-string matching. Query params should be their own retrieval channel so verbose boilerplate does not dilute the short query.")

    A("\n## 6. Lexical coverage on known positives\n")
    A(f"This section uses a deterministic sample of {len(val):,} train positives whose item is still in the benchmark corpus.\n")
    A(md_table([[c, pct(cov[c].mean())] for c in cov_cols], ["condition", "positive pairs"]))
    A("")
    A("Benchmark query-token occurrence OOV rate against corpus fields:\n")
    A(md_table([[k, pct(v)] for k, v in oov_stats.items()], ["corpus field", "token-occurrence OOV"]))
    A(f"- Queries with zero token overlap against title+params+description vocabulary: {pct(bench_zero_union)}.")
    A("- Title is a precision channel; description supplies substantial extra recall. Params are valuable but must be fielded. Character retrieval remains important for spelling/morphology/transliteration.")

    A("\n## 7. Item popularity and duplication\n")
    A(f"- Unique clicked train items: {len(click_counts_all):,}; click count quantiles: {quantiles(click_counts_all)}.")
    A(f"- Corpus items with at least one historical positive: {len(click_counts_bench):,} ({pct(len(click_counts_bench)/len(items))}).")
    A("\nHistorical click mass captured by the most-clicked fraction of items:\n")
    A(md_table([[f"top {100*f:g}%", pct(v)] for f, v in head_share.items()], ["item head", "share of clicks"]))
    A("")
    A(f"- Duplicate normalized-title groups in corpus: {len(duplicate_title_groups):,}; rows in such groups: {duplicate_title_rows:,} ({pct(duplicate_title_rows/len(items))}).")
    A(f"- Rows participating in exact raw title+params+description duplicates: {duplicated_content_rows:,} ({pct(duplicated_content_rows/len(items))}).")
    A(f"- Train IDs with changing title/microcat/location/category across rows: title={(id_consistency.titles>1).sum():,}, microcat={(id_consistency.microcats>1).sum():,}, location={(id_consistency.locations>1).sum():,}, category={(id_consistency.categories>1).sum():,}.")
    A("\nLargest duplicate-title groups:\n")
    A(broad_title_examples.to_markdown(index=False))
    A("- Near-duplicate listings can waste the 50-slot recall budget. A diversity pass across normalized title/content clusters is worth validating, but should preserve historically clicked exact IDs.")

    A("\n## 8. Concrete modeling hypotheses\n")
    hypotheses = [
        "**Hierarchical history retrieval:** exact normalized full context `(query, location, delivery, params, category)` → `(query, location)` → query-only. Give each level its own RRF list instead of one query-only click list.",
        "**Conditional geography:** estimate `P(item_location=query_location | microcat/query cluster)` from positives. Replace the global location bonus with log-odds; turn it down for remote/online wording and microcats with low same-location rates.",
        "**Microcategory posterior, not top-3 set:** use smoothed `P(microcat|query)` and confidence/entropy. Strong boost only for low-entropy known queries; for ambiguous queries retrieve separately within several microcats and quota-fuse.",
        "**Fielded hybrid:** separate BM25/TF-IDF channels for title, description, item params, and exact filter tokens. Fuse ranks rather than concatenating long fields. Short queries particularly benefit from title+char; detailed queries from description.",
        "**Pseudo-document expansion without leakage:** append historically clicked titles, params values, and microcategory-specific salient terms to known queries, but build validation profiles from disjoint query events/items.",
        "**Popularity as a conditional tie-breaker:** global popularity is unsafe; use click popularity within `(normalized query, location/microcat)` or review quality within a semantic candidate pool.",
        "**Duplicate-aware fill:** after preserving top high-confidence items, cap near-identical title/content clusters and spend freed slots on alternate microcats/lexical channels. Evaluate only on multi-positive groups because dedupe can hurt when duplicate IDs are separately relevant.",
        "**Gated ensembles:** known low-entropy queries, ambiguous known queries, unseen queries, nonempty filters, and remote intent should have different fusion weights. Aggregate Recall hides these radically different regimes.",
        "**Hard-negative reranker:** train a lightweight pairwise/logistic or LightGBM model on positives versus same-query lexical hard negatives. Features: field similarities, full/query/location history, microcat posterior, same-location prior, param-token matches, ratings/reviews, remote flags.",
        "**Validation fix:** split by time is unavailable; use group-based holdout and remove held-out item interactions from all history profiles. Report Recall by known/unseen query and by ambiguity so history gains are not mistaken for semantic generalization.",
    ]
    for h in hypotheses:
        A(f"- {h}")

    A("\n## 9. Highest-priority experiments\n")
    A("1. Add full-context and `(query, location)` history lists as separate RRF channels; measure known/unseen slices.")
    A("2. Tune conditional location priors by microcategory and remote intent; this directly attacks the largest structured signal.")
    A("3. Replace concatenated word TF-IDF with fielded title/description/params retrieval plus RRF.")
    A("4. Use posterior-weighted microcategory retrieval with entropy gating rather than a flat top-3 bonus.")
    A("5. Train a simple hard-negative reranker and compare to hand-set bonuses.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
