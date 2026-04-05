import pandas as pd


def load_ciliary_ensembl_ids(csv_path='CiliaCarta.csv'):
    """Load ciliary Ensembl Gene IDs from CiliaCarta.csv."""
    table = pd.read_csv(csv_path)
    if 'Ensembl Gene ID' not in table.columns:
        raise ValueError("'Ensembl Gene ID' column not found in CiliaCarta.csv")
    return [
        str(gene).strip()
        for gene in table['Ensembl Gene ID'].tolist()
        if pd.notna(gene) and str(gene).strip() and str(gene).strip() != 'ENSG00000161574'
    ]
