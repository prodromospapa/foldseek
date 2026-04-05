import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from tqdm import tqdm
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

EVALUE_THRESHOLD     = 1e-5   # keep hits with evalue <= this
DEDUP_REGION_OVERLAP = 0.8    # fraction of the SHORTER interval that must overlap
                               # for two hits to the SAME query–target pair to be
                               # considered redundant (best-evalue hit is kept)
MIN_REGION_OVERLAP   = 0.8    # fraction of the SHORTER interval that must overlap
                               # for two hits to the same target to count as shared
                               # across queries (used in Jaccard computation)
CLUSTER_CUTOFF       = 0.5    # fcluster distance cut-off
MIN_CLUSTER_SIZE     = 5      # skip boundary boxes for clusters smaller than this

# NOTE: delete cached CSVs if you change EVALUE_THRESHOLD, DEDUP_REGION_OVERLAP,
#       or MIN_REGION_OVERLAP, as the cached matrices will be stale.

# Presentation output settings
FIGURE_WIDTH  = 16
FIGURE_HEIGHT = 13
OUTPUT_DPI    = 180
FONT_TITLE    = 16
FONT_TICK     = 9
FONT_CBAR     = 12
FONT_ANNOT    = 7


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_pickle('foldseek_combined_results_with_info.pkl')


# ── Filter ────────────────────────────────────────────────────────────────────

working = df.copy()
if EVALUE_THRESHOLD is not None and 'evalue' in working.columns:
    before  = len(working)
    working = working[working['evalue'] <= EVALUE_THRESHOLD]
    print(f"E-value filter: {before:,} → {len(working):,} rows (evalue ≤ {EVALUE_THRESHOLD:.0e})")
else:
    print(f"No e-value filter applied. Columns: {list(working.columns)}")

# Check required columns
for col in ['tstart', 'tend', 'evalue']:
    if col not in working.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(working.columns)}")


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_hits(df_in, query_col, target_col='target_uniprot_id',
                     tstart_col='tstart', tend_col='tend', evalue_col='evalue',
                     overlap_threshold=DEDUP_REGION_OVERLAP):
    """
    For each (query, target) pair, greedily retain non-redundant hits.

    Hits are processed in ascending e-value order (best first). A hit is
    discarded when it overlaps >= overlap_threshold of the SHORTER of itself
    and any already-kept hit for the same pair.

    Parameters
    ----------
    df_in             : DataFrame after e-value filtering
    query_col         : column identifying the query entity (UniProt or Ensembl)
    target_col        : column identifying the target UniProt accession
    tstart_col/tend_col : target region start / end
    evalue_col        : e-value column used to rank hits (lower = better)
    overlap_threshold : minimum overlap fraction to call two hits redundant

    Returns
    -------
    Deduplicated DataFrame (same columns, subset of rows).
    """
    mask   = df_in[query_col].notna() & df_in[target_col].notna()
    df_sub = df_in[mask].copy()
    other  = df_in[~mask].copy()   # rows without keys pass through unchanged

    df_sub = df_sub.sort_values(evalue_col, ascending=True)

    q_arr = df_sub[query_col].astype(str).values
    t_arr = df_sub[target_col].astype(str).values
    s_arr = df_sub[tstart_col].values.astype(float)
    e_arr = df_sub[tend_col].values.astype(float)

    pair_to_idx = defaultdict(list)
    for i, (q, t) in enumerate(zip(q_arr, t_arr)):
        pair_to_idx[(q, t)].append(i)

    keep_flags = np.zeros(len(df_sub), dtype=bool)

    for indices in pair_to_idx.values():
        kept_intervals = []
        for i in indices:
            s, e = s_arr[i], e_arr[i]
            span = e - s
            if span <= 0:
                keep_flags[i] = True
                continue

            redundant = False
            for (ks, ke) in kept_intervals:
                kspan = ke - ks
                if kspan <= 0:
                    continue
                overlap = max(0.0, min(e, ke) - max(s, ks))
                if overlap / min(span, kspan) >= overlap_threshold:
                    redundant = True
                    break

            if not redundant:
                kept_intervals.append((s, e))
                keep_flags[i] = True

    kept_df = df_sub[keep_flags]
    result  = pd.concat([kept_df, other], ignore_index=True)
    removed = len(df_sub) - keep_flags.sum()
    print(f"    Dedup [{query_col}]: {len(df_sub):,} → {keep_flags.sum():,} rows "
          f"({removed:,} redundant hits removed, overlap ≥ {overlap_threshold})")
    return result


# ── Region-aware helpers ──────────────────────────────────────────────────────

def build_homologue_regions(df_subset, query_col,
                             target_col='target_uniprot_id',
                             tstart_col='tstart', tend_col='tend'):
    regions = defaultdict(lambda: defaultdict(list))

    mask   = df_subset[query_col].notna() & df_subset[target_col].notna()
    df_sub = df_subset[mask].dropna(subset=[tstart_col, tend_col])

    q_vals = df_sub[query_col].astype(str).values
    t_vals = df_sub[target_col].astype(str).values
    s_vals = df_sub[tstart_col].values.astype(int)
    e_vals = df_sub[tend_col].values.astype(int)

    for q, t, s, e in zip(q_vals, t_vals, s_vals, e_vals):
        regions[q][t].append((s, e))

    return {q: dict(td) for q, td in regions.items()}


def any_interval_overlaps(intervals_a, intervals_b, min_overlap):
    """
    Return True if ANY interval from A overlaps with ANY interval from B
    by >= min_overlap fraction of the SHORTER of the two intervals.
    """
    for (a_s, a_e) in intervals_a:
        len_a = a_e - a_s
        if len_a <= 0:
            continue
        for (b_s, b_e) in intervals_b:
            len_b = b_e - b_s
            if len_b <= 0:
                continue
            overlap  = max(0, min(a_e, b_e) - max(a_s, b_s))
            min_len  = min(len_a, len_b)
            if overlap / min_len >= min_overlap:
                return True
    return False


def jaccard_region_aware(regions_a, regions_b, min_overlap):
    """
    Jaccard similarity between two region dicts.

    intersection = targets in BOTH with sufficient regional overlap
    union        = |A| + |B| - intersection
    """
    targets_a = set(regions_a)
    targets_b = set(regions_b)
    shared    = targets_a & targets_b

    inter = sum(
        1 for t in shared
        if any_interval_overlaps(regions_a[t], regions_b[t], min_overlap)
    )
    union = len(targets_a) + len(targets_b) - inter
    return (inter / union if union > 0 else 0.0), inter


# ── Jaccard matrix ────────────────────────────────────────────────────────────

def build_jaccard(gene_region_dicts, label=''):
    genes = sorted(gene_region_dicts.keys())
    n     = len(genes)
    jac   = np.zeros((n, n))
    ovl   = np.zeros((n, n), dtype=int)

    with tqdm(total=n, desc=f'Jaccard [{label}]', unit='gene', dynamic_ncols=True) as pbar:
        for i, g1 in enumerate(genes):
            jac[i, i] = 1.0
            ovl[i, i] = len(gene_region_dicts[g1])
            for j, g2 in enumerate(genes):
                if j <= i:
                    continue
                j_val, inter = jaccard_region_aware(
                    gene_region_dicts[g1], gene_region_dicts[g2], MIN_REGION_OVERLAP
                )
                jac[i, j] = jac[j, i] = j_val
                ovl[i, j] = ovl[j, i] = inter
            pbar.update(1)

    dist = 1 - jac
    np.fill_diagonal(dist, 0)
    Z     = linkage(squareform(dist, checks=False), method='complete')
    order = leaves_list(Z)
    ids   = fcluster(Z, t=CLUSTER_CUTOFF, criterion='distance')

    return (
        jac[np.ix_(order, order)],
        ovl[np.ix_(order, order)],
        [genes[i] for i in order],
        ids[order],
    )


def cluster_ids_from_matrix(jac_matrix):
    dist = 1 - jac_matrix
    np.fill_diagonal(dist, 0)
    # Use 'complete' linkage to match build_jaccard
    Z = linkage(squareform(dist, checks=False), method='complete')
    return fcluster(Z, t=CLUSTER_CUTOFF, criterion='distance')


def load_or_compute(jac_csv, ovl_csv, gene_region_dicts, label):
    """Load from CSV cache if available; otherwise compute and save.
    NOTE: delete cached CSVs if you change EVALUE_THRESHOLD, DEDUP_REGION_OVERLAP,
    or MIN_REGION_OVERLAP."""
    if Path(jac_csv).exists() and Path(ovl_csv).exists():
        print(f"  Loading cached tables: {jac_csv}, {ovl_csv}")
        jac_df  = pd.read_csv(jac_csv, index_col=0)
        ovl_df  = pd.read_csv(ovl_csv, index_col=0)
        labels  = list(jac_df.index)
        jac_ord = jac_df.values
        ovl_ord = ovl_df.values.astype(int)
        ids_ord = cluster_ids_from_matrix(jac_ord)
        return jac_ord, ovl_ord, labels, ids_ord
    else:
        print(f"  Cache not found — computing [{label}]...")
        jac_ord, ovl_ord, labels, ids_ord = build_jaccard(gene_region_dicts, label=label)
        pd.DataFrame(jac_ord, index=labels, columns=labels).to_csv(jac_csv)
        pd.DataFrame(ovl_ord, index=labels, columns=labels).to_csv(ovl_csv)
        print(f"  Saved → {jac_csv}, {ovl_csv}")
        return jac_ord, ovl_ord, labels, ids_ord


# ── Heatmap ───────────────────────────────────────────────────────────────────

def get_cluster_blocks(ids):
    blocks, start = [], 0
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            blocks.append((start, i - 1))
            start = i
    blocks.append((start, len(ids) - 1))
    return blocks


def export_clusters_to_csv(labels, ids, output_file):
    """Export cluster assignments to a CSV file."""
    cluster_df = pd.DataFrame({'gene': labels, 'cluster': ids})
    cluster_sizes    = cluster_df['cluster'].value_counts()
    large_clusters   = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index
    cluster_filtered = cluster_df[cluster_df['cluster'].isin(large_clusters)]
    cluster_sorted   = cluster_filtered.sort_values(['cluster', 'gene'])
    cluster_sorted.to_csv(output_file, index=False)
    print(f"  Saved cluster assignments (≥{MIN_CLUSTER_SIZE}) → {output_file}")


def make_heatmap(jac_ord, ovl_ord, labels, ids_ord, title, output_file):
    n   = len(labels)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    hm = sns.heatmap(
        jac_ord, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cmap='Blues', vmin=0, vmax=1, linewidths=0,
        annot=(n <= 25), fmt='.2f', annot_kws={'size': FONT_ANNOT},
        cbar_kws={'label': 'Jaccard similarity', 'shrink': 0.6, 'ticks': cbar_ticks},
    )

    cbar = hm.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    cbar.ax.tick_params(labelsize=FONT_CBAR)
    cbar.set_label('Jaccard similarity', fontsize=FONT_CBAR, labelpad=10)

    blocks = get_cluster_blocks(ids_ord)
    drawn  = 0
    for (start, end) in blocks:
        size = end - start + 1
        if size < MIN_CLUSTER_SIZE:
            continue
        ax.add_patch(mpatches.Rectangle(
            (start, start), size, size,
            fill=False, edgecolor='black', linewidth=2.0, clip_on=False,
        ))
        drawn += 1

    ax.set_title(
        f"{title}\n{drawn} cluster(s) shown (≥{MIN_CLUSTER_SIZE} genes, "
        f"d ≤ {CLUSTER_CUTOFF}, region overlap ≥ {MIN_REGION_OVERLAP})",
        fontsize=FONT_TITLE, pad=16, fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=FONT_TICK, rotation=45)
    ax.tick_params(axis='y', labelsize=FONT_TICK, rotation=0)

    plt.tight_layout()
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight',
                pil_kwargs={'optimize': True})
    plt.close(fig)
    print(f"  Saved → {output_file}")


# ── Heatmap 1: UniProt ID level ───────────────────────────────────────────────

print("\n── UniProt-level heatmap ──")
print(f"  Deduplicating by query_uniprot_id × target_uniprot_id ...")
working_dedup_uniprot = deduplicate_hits(
    working, query_col='query_uniprot_id',
    overlap_threshold=DEDUP_REGION_OVERLAP,
)

homologue_regions_uniprot = build_homologue_regions(
    working_dedup_uniprot, query_col='query_uniprot_id'
)
print(f"  {len(homologue_regions_uniprot):,} UniProt accessions")

jac1, ovl1, labels1, ids1 = load_or_compute(
    'jaccard_uniprot.csv', 'overlap_counts_uniprot.csv',
    homologue_regions_uniprot, 'UniProt',
)
make_heatmap(
    jac1, ovl1, labels1, ids1,
    title=f"Homologue Overlap — UniProt level (n={len(labels1)}, evalue ≤ {EVALUE_THRESHOLD:.0e})",
    output_file='heatmap_uniprot.png',
)

export_clusters_to_csv(labels1, ids1, 'clusters_uniprot.csv')


# ── Heatmap 2: Gene level (grouped by query_ensembl_id) ──────────────────────

print("\n── Gene-level heatmap ──")
working['query_protein_name'] = working['query_protein_name'].replace('', np.nan)

ensembl_label = (
    working.dropna(subset=['query_ensembl_id'])
    .query("query_ensembl_id != ''")
    .groupby('query_ensembl_id')['query_protein_name']
    .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else None)
)

has_ensembl  = working['query_ensembl_id'].notna() & (working['query_ensembl_id'] != '')
w_ensembl    = working[has_ensembl].copy()
w_no_ensembl = working[~has_ensembl].copy()

print(f"  Deduplicating Ensembl-keyed rows ...")
w_ensembl_dedup = deduplicate_hits(
    w_ensembl, query_col='query_ensembl_id',
    overlap_threshold=DEDUP_REGION_OVERLAP,
)

print(f"  Deduplicating UniProt-fallback rows ...")
w_no_ensembl_dedup = deduplicate_hits(
    w_no_ensembl, query_col='query_uniprot_id',
    overlap_threshold=DEDUP_REGION_OVERLAP,
)

regions_ensembl = build_homologue_regions(w_ensembl_dedup,    query_col='query_ensembl_id')
regions_uniprot = build_homologue_regions(w_no_ensembl_dedup, query_col='query_uniprot_id')

assert not (set(regions_ensembl) & set(regions_uniprot)), (
    "Key collision between Ensembl and UniProt region dicts — "
    "check for shared IDs across the two namespaces."
)

homologue_regions_gene = {**regions_ensembl, **regions_uniprot}

display_labels = {eid: (ensembl_label.get(eid) or eid) for eid in homologue_regions_gene}
name_counts    = Counter(display_labels.values())
homologue_regions_display = {
    (f"{name} ({eid})" if name_counts[name] > 1 else name): homologue_regions_gene[eid]
    for eid, name in display_labels.items()
}
print(f"  {len(homologue_regions_display):,} genes (Ensembl IDs + UniProt fallbacks)")

jac2, ovl2, labels2, ids2 = load_or_compute(
    'jaccard_genes.csv', 'overlap_counts_genes.csv',
    homologue_regions_display, 'Gene',
)
make_heatmap(
    jac2, ovl2, labels2, ids2,
    title=f"Homologue Overlap — Gene level (n={len(labels2)}, evalue ≤ {EVALUE_THRESHOLD:.0e})",
    output_file='heatmap_genes.png',
)

export_clusters_to_csv(labels2, ids2, 'clusters_genes.csv')

print("\nDone.")