import ast
import os
import pandas as pd
import pickle
import tqdm
from collections import defaultdict
from ciliary_genes import load_ciliary_ensembl_ids


def _select_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

with open('subcell_hierarchy.pkl', 'rb') as f:
    tree = pickle.load(f)

# ── Accession → organism lookup (built from Swiss-Prot XML, cached) ───────────

ACC_ORG_CACHE = 'swissprot_accession_organism.pkl'

if not os.path.exists(ACC_ORG_CACHE):
    raise FileNotFoundError(
        f'{ACC_ORG_CACHE} not found — please run 7.1.compute_swissprot_counts.py first.'
    )

with open(ACC_ORG_CACHE, 'rb') as f:
    accession_organism = pickle.load(f)
print(f'Loaded accession→organism lookup ({len(accession_organism):,} accessions)')

print("Loading foldseek results...")
dataframe = pd.read_pickle("foldseek_combined_results_with_info.pkl")
print(f"  {len(dataframe):,} rows loaded. Filtering evalue <= 1e-3...")
dataframe = dataframe[dataframe["evalue"] <= 1e-3]
print(f"  {len(dataframe):,} rows after filter.")

def _is_empty(val):
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, (list, tuple, set)):
        return len(val) == 0
    if isinstance(val, str):
        return val.strip() == '' or val.strip() in ('[]', 'nan', 'None')
    return False

has_ancestors = dataframe["target_localization_ancestors"].apply(lambda x: not _is_empty(x)) \
    if "target_localization_ancestors" in dataframe.columns else pd.Series(False, index=dataframe.index)
has_uniprot = dataframe["target_localization_uniprot"].apply(lambda x: not _is_empty(x)) \
    if "target_localization_uniprot" in dataframe.columns else pd.Series(False, index=dataframe.index)

dataframe = dataframe[has_ancestors | has_uniprot]
print(f"  {len(dataframe):,} rows after removing entries with no localization (ancestors or uniprot).")

query_col = _select_column(dataframe, ["query_ensembl_id", "query_uniprot_id", "query_gene_name"])
target_uniprot_col = _select_column(dataframe, ["target_uniprot_id", "target_gene_name"])
loc_col = _select_column(dataframe, ["target_localization_ancestors", "target_localization_uniprot"])

if query_col is None:
    raise RuntimeError("query column not found in foldseek results")
if target_uniprot_col is None:
    raise RuntimeError("target identifier column not found in foldseek results")
if loc_col is None:
    raise RuntimeError("target localization column not found in foldseek results")

ciliary_ids = load_ciliary_ensembl_ids()
present_queries = set(dataframe[query_col].dropna().astype(str).unique())
queries = [q for q in ciliary_ids if q in present_queries]
print(f"  {len(queries):,} ciliary queries present in foldseek results.")

per_query_counts = {q: defaultdict(int) for q in queries}
compartments = [key for key in tree.keys() if key != 'All proteins']

grouped = dataframe.groupby(dataframe[query_col].astype(str))
for query in tqdm.tqdm(
    queries,
    total=len(queries),
    desc="Counting per-query compartment hits",
    dynamic_ncols=True,
):
    try:
        group = grouped.get_group(query)
    except KeyError:
        continue

    seen_targets = set()
    for _, row in group.iterrows():
        target_id = row.get(target_uniprot_col)
        if pd.isna(target_id):
            continue
        target_key = str(target_id).strip()
        if not target_key or target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        locs = row.get(loc_col) or []
        if isinstance(locs, str):
            try:
                locs = ast.literal_eval(locs)
            except Exception:
                locs = [locs]
        if not isinstance(locs, (list, tuple, set)):
            locs = [locs]

        seen_anc = set()
        for loc in locs:
            if not loc:
                continue
            loc_key = str(loc).strip().lower()
            if loc_key not in tree:
                continue
            for ancestor in tree[loc_key]:
                if ancestor in seen_anc:
                    continue
                seen_anc.add(ancestor)
                if ancestor != 'All proteins':
                    per_query_counts[query][ancestor] += 1

per_query_df = pd.DataFrame.from_dict(per_query_counts, orient='index')
per_query_df = per_query_df.reindex(index=queries, columns=compartments, fill_value=0).fillna(0).astype(int)
per_query_df.index.name = 'Query'
per_query_df['Total'] = per_query_df[compartments].sum(axis=1)
per_query_df = per_query_df[[*compartments, 'Total']]
per_query_df.to_csv('per_query_organelle_counts_proteins.csv')
print('Saved per_query_organelle_counts_proteins.csv')

# Count proteins and genes per organelle for three localization sources independently
LOC_SOURCES = [
    ('uniprot',   'target_localization_uniprot'),
    ('go',        'target_localization_ancestors'),
    ('combined',  None),  # union of uniprot + go
]

for source_name, loc_col_name in LOC_SOURCES:
    count_tree_unique        = {key: 0 for key in tree.keys()}
    count_tree_species       = defaultdict(lambda: defaultdict(int))
    count_tree_genes         = {key: 0 for key in tree.keys()}
    count_tree_genes_species = defaultdict(lambda: defaultdict(int))

    seen_proteins = set()
    seen_genes    = set()

    if source_name == 'combined':
        uniprot_col = 'target_localization_uniprot'   if 'target_localization_uniprot'   in dataframe.columns else None
        go_col      = 'target_localization_ancestors' if 'target_localization_ancestors' in dataframe.columns else None
        iter_cols   = ["target_uniprot_id", "target_gene_name"]
        if uniprot_col: iter_cols.append(uniprot_col)
        if go_col:      iter_cols.append(go_col)
    else:
        if loc_col_name not in dataframe.columns:
            print(f"  Skipping '{source_name}': column '{loc_col_name}' not found.")
            continue
        iter_cols = ["target_uniprot_id", "target_gene_name", loc_col_name]

    desc = f"Counting proteins/genes per organelle [{source_name}]"
    for row_tuple in tqdm.tqdm(
        dataframe[iter_cols].itertuples(index=False, name=None),
        total=len(dataframe),
        desc=desc,
        dynamic_ncols=True,
    ):
        if source_name == 'combined':
            uniprot_id = row_tuple[0]
            gene_name  = row_tuple[1]
            locs_uniprot = row_tuple[2] if uniprot_col else []
            locs_go      = row_tuple[3] if (uniprot_col and go_col) else (row_tuple[2] if go_col else [])
            locations = list({loc for locs in (locs_uniprot or [], locs_go or []) if isinstance(locs, list) for loc in locs})
        else:
            uniprot_id, gene_name, locations = row_tuple

        organism       = accession_organism.get(uniprot_id, '')
        is_new_protein = uniprot_id not in seen_proteins
        gene_key       = (gene_name, organism) if (isinstance(gene_name, str) and gene_name.strip()) else None
        is_new_gene    = gene_key is not None and gene_key not in seen_genes

        if not is_new_protein and not is_new_gene:
            continue

        if is_new_protein:
            seen_proteins.add(uniprot_id)
        if is_new_gene:
            seen_genes.add(gene_key)

        if not isinstance(locations, list):
            continue

        seen_anc_p = set()
        seen_anc_g = set()
        for location in locations:
            if location and location in tree:
                for ancestor in tree[location]:
                    if is_new_protein and ancestor not in seen_anc_p:
                        seen_anc_p.add(ancestor)
                        count_tree_unique[ancestor] += 1
                        if organism:
                            count_tree_species[ancestor][organism] += 1
                    if is_new_gene and ancestor not in seen_anc_g:
                        seen_anc_g.add(ancestor)
                        count_tree_genes[ancestor] += 1
                        if organism:
                            count_tree_genes_species[ancestor][organism] += 1

    print(f"  [{source_name}] {len(seen_proteins):,} unique proteins, {len(seen_genes):,} unique genes.")

    # ── Save protein counts ───────────────────────────────────────────────────
    count_df_species = pd.DataFrame.from_dict(count_tree_species, orient="index")
    count_df_species = count_df_species.reindex(tree.keys(), fill_value=0).fillna(0).astype(int)
    count_df_species.index.name = "Organelle"
    count_df_species["Total"] = pd.Series(count_tree_unique)
    count_df_species = count_df_species[count_df_species["Total"] > 0]
    count_df_species = count_df_species.sort_values(by="Total", ascending=False)
    proteins_csv = f"organelle_counts_proteins_{source_name}.csv"
    count_df_species.reset_index().to_csv(proteins_csv, index=False)
    print(f"Saved {proteins_csv}")

    # ── Save gene counts ──────────────────────────────────────────────────────
    count_df_genes = pd.DataFrame.from_dict(count_tree_genes_species, orient="index")
    count_df_genes = count_df_genes.reindex(tree.keys(), fill_value=0).fillna(0).astype(int)
    count_df_genes.index.name = "Organelle"
    count_df_genes["Total"] = pd.Series(count_tree_genes)
    count_df_genes = count_df_genes[count_df_genes["Total"] > 0]
    count_df_genes = count_df_genes.sort_values(by="Total", ascending=False)
    genes_csv = f"organelle_counts_genes_{source_name}.csv"
    count_df_genes.reset_index().to_csv(genes_csv, index=False)
    print(f"Saved {genes_csv}")