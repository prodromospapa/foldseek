"""
Compute per-compartment per-species protein and gene counts for Swiss-Prot,
TrEMBL, and combined UniProtKB, reading from uniprot_index.db (built by
6.uniprot_full.py).  No XML download needed.

The `reviewed` column distinguishes Swiss-Prot (1) from TrEMBL (0).
The `organism` column and `localization_ancestors` / `gene_name` fields come
from the stored data blobs.

Outputs (same filenames as before):
    swissprot_compartment_counts_proteins_species.pkl
    trembl_compartment_counts_proteins_species.pkl
    uniprot_compartment_counts_proteins_species.pkl

    swissprot_compartment_counts_genes_species.pkl
    trembl_compartment_counts_genes_species.pkl
    uniprot_compartment_counts_genes_species.pkl

    swissprot_accession_organism.pkl
    trembl_accession_organism.pkl
    uniprot_accession_organism.pkl

Usage:
    python3 7.1.compute_uniprot_counts.py
"""

import pickle
import sqlite3
import sys
from collections import defaultdict

from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

UNIPROT_DB_FILE = 'uniprot_index.db'

PROTEIN_CACHE_FILES = {
    'swissprot': 'swissprot_compartment_counts_proteins_species.pkl',
    'trembl':    'trembl_compartment_counts_proteins_species.pkl',
    'combined':  'uniprot_compartment_counts_proteins_species.pkl',
}

GENE_CACHE_FILES = {
    'swissprot': 'swissprot_compartment_counts_genes_species.pkl',
    'trembl':    'trembl_compartment_counts_genes_species.pkl',
    'combined':  'uniprot_compartment_counts_genes_species.pkl',
}

ACC_ORG_CACHE_FILES = {
    'swissprot': 'swissprot_accession_organism.pkl',
    'trembl':    'trembl_accession_organism.pkl',
    'combined':  'uniprot_accession_organism.pkl',
}

# ── Count accumulators ────────────────────────────────────────────────────────

species_protein_counts = {
    'swissprot': defaultdict(lambda: defaultdict(int)),
    'trembl':    defaultdict(lambda: defaultdict(int)),
    'combined':  defaultdict(lambda: defaultdict(int)),
}

species_gene_sets = {
    'swissprot': defaultdict(lambda: defaultdict(set)),
    'trembl':    defaultdict(lambda: defaultdict(set)),
    'combined':  defaultdict(lambda: defaultdict(set)),
}

accession_organism = {
    'swissprot': {},
    'trembl':    {},
    'combined':  {},
}

protein_counts = {'swissprot': 0, 'trembl': 0}

# ── Stream DB ─────────────────────────────────────────────────────────────────

print(f'Opening {UNIPROT_DB_FILE} ...', file=sys.stderr)
conn = sqlite3.connect(UNIPROT_DB_FILE)

total = conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
print(f'  {total:,} entries', file=sys.stderr)

rows = conn.execute('SELECT accession, reviewed, organism, data FROM entries')

for accession, reviewed, organism, data in tqdm(rows, total=total, unit='entry', file=sys.stderr):
    source = 'swissprot' if reviewed else 'trembl'

    parsed = pickle.loads(data)
    locs      = parsed.get('localization_ancestors') or []
    gene_name = (parsed.get('gene_name') or '').strip().lower()
    organism  = (organism or '').strip().lower()

    if organism:
        accession_organism[source][accession]     = organism
        accession_organism['combined'][accession] = organism

    if not organism or not locs:
        protein_counts[source] += 1
        continue

    seen_p = set()
    seen_g = set()
    for loc in locs:
        if not loc:
            continue
        if loc not in seen_p:
            seen_p.add(loc)
            species_protein_counts[source]['combined'][loc]    += 1  # unused but harmless
            species_protein_counts[source][organism][loc]      += 1
            species_protein_counts['combined'][organism][loc]  += 1
        if gene_name and loc not in seen_g:
            seen_g.add(loc)
            species_gene_sets[source][organism][loc].add(gene_name)
            species_gene_sets['combined'][organism][loc].add(gene_name)

    protein_counts[source] += 1

conn.close()

# ── Save caches ───────────────────────────────────────────────────────────────

for key in ('swissprot', 'trembl', 'combined'):
    plain_proteins = {
        species: dict(comp_dict)
        for species, comp_dict in species_protein_counts[key].items()
        if species != 'combined'
    }
    with open(PROTEIN_CACHE_FILES[key], 'wb') as f:
        pickle.dump(plain_proteins, f)
    print(f'Saved protein counts to {PROTEIN_CACHE_FILES[key]} ({len(plain_proteins)} species)')

for key in ('swissprot', 'trembl', 'combined'):
    plain_genes = {
        species: {comp: len(genes) for comp, genes in comp_dict.items()}
        for species, comp_dict in species_gene_sets[key].items()
    }
    with open(GENE_CACHE_FILES[key], 'wb') as f:
        pickle.dump(plain_genes, f)
    print(f'Saved gene counts to {GENE_CACHE_FILES[key]} ({len(plain_genes)} species)')

for key in ('swissprot', 'trembl', 'combined'):
    with open(ACC_ORG_CACHE_FILES[key], 'wb') as f:
        pickle.dump(accession_organism[key], f)
    print(f'Saved accession→organism to {ACC_ORG_CACHE_FILES[key]} ({len(accession_organism[key]):,} accessions)')

# ── Summary ───────────────────────────────────────────────────────────────────

print('\nDone.')
print(f'Swiss-Prot proteins: {protein_counts["swissprot"]:,}')
print(f'TrEMBL proteins:     {protein_counts["trembl"]:,}')
print(f'Combined:            {protein_counts["swissprot"] + protein_counts["trembl"]:,}')
