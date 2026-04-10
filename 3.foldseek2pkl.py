import os
import re
import tarfile
import numpy as np
import pandas as pd
import asyncio
import aiohttp
from tqdm import tqdm

# ── Column definitions ────────────────────────────────────────────────────────
columns = [
    'target_model',
    'pident',
    'alnlen',
    'mismatch',
    'gapopen',
    'qstart',
    'qend',
    'tstart',
    'tend',
    'prob',
    'evalue',
    'query_sequence',
    'target_sequence',
    'taxon_id',
    'species',
]

RESULT_COLS = [
    "target_uniprot_id", "target_chain_id", "target_isoform",
    "target_dimer_uniprot_id", "target_dimer_isoform",
    "target_uniprot_reviewed", "target_dimer_uniprot_reviewed",
]

_FALLBACK = (np.nan, np.nan, np.nan, np.nan, np.nan, False, False)

# Max Ensembl IDs processed concurrently. Tune to balance speed vs. API rate limits.
ENSEMBL_CONCURRENCY = 5

# ── FIX 5: Pre-index query metadata for O(1) lookup (was O(n) scan per file) ─
query_info = pd.read_csv("ciliary_structures.csv")
query_info_indexed = query_info.set_index("ensembl_id")

# ── FIX 1 & 2: Proper async-safe caches using Task/Future pattern ─────────────
# Stores asyncio.Task objects so the same HTTP call is never duplicated,
# even when many coroutines request the same ID concurrently.
_uniprot_task_cache: dict = {}   # base_uniprot_id  → asyncio.Task[(iso_num, reviewed)]
_pdb_task_cache: dict = {}       # (pdb_id, chain_id) → asyncio.Task[(uniprot, isoform, reviewed)]


# ── UniProt helper ─────────────────────────────────────────────────────────────

async def _fetch_uniprot_info(session: aiohttp.ClientSession, base: str) -> tuple:
    """
    FIX 4: Single JSON request replaces the original txt + JSON fallback.
    The .json endpoint is structured and smaller than the full flat-text file.
    """
    iso_num, reviewed = "1", False
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{base}.json"
        async with session.get(url, headers={"User-Agent": "foldseek-script/1.0"}) as resp:
            if resp.status == 200:
                data = await resp.json()
                reviewed = data.get("entryType", "").lower() == "reviewed"
                isoforms = data.get("isoforms", [])
                if isoforms:
                    first_id = isoforms[0].get("isoformIds", [""])[0]
                    parts = first_id.split("-")
                    if len(parts) > 1 and parts[1].isdigit():
                        iso_num = parts[1]
    except Exception:
        pass
    return iso_num, reviewed


def uniprot_info_task(session: aiohttp.ClientSession, uniprot_id) -> asyncio.Task:
    """
    FIX 1: Returns a shared Task for each unique base ID.
    Multiple coroutines awaiting the same ID share one HTTP request.
    """
    if not uniprot_id or (isinstance(uniprot_id, float) and np.isnan(uniprot_id)):
        async def _default():
            return "1", False
        return asyncio.ensure_future(_default())

    base = str(uniprot_id).split("-")[0]
    if base not in _uniprot_task_cache:
        _uniprot_task_cache[base] = asyncio.ensure_future(
            _fetch_uniprot_info(session, base)
        )
    return _uniprot_task_cache[base]


async def uniprot_info(session: aiohttp.ClientSession, uniprot_id) -> tuple:
    return await uniprot_info_task(session, uniprot_id)


# ── PDB → UniProt ──────────────────────────────────────────────────────────────

async def _fetch_pdb2uniprot(
    session: aiohttp.ClientSession, pdb_id: str, chain_id: str
) -> tuple:
    """
    FIX 2: Uses the shared session passed in — no more new ClientSession per call.
    """
    try:
        async with session.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        ) as resp:
            entry_json = await resp.json()

        polymer_ids = (
            entry_json
            .get("rcsb_entry_container_identifiers", {})
            .get("polymer_entity_ids", [])
        )
        for entity_id in polymer_ids:
            async with session.get(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
            ) as resp:
                entity_json = await resp.json()

            chains = (
                entity_json.get("entity_poly", {})
                .get("pdbx_strand_id", "")
                .split(",")
            )
            if chain_id not in chains:
                continue

            identifiers = entity_json.get("rcsb_polymer_entity_container_identifiers", {})
            uniprot_id = (
                identifiers["uniprot_ids"][0]
                if identifiers.get("uniprot_ids") else None
            )
            isoform = None
            for ref in identifiers.get("reference_sequence_identifiers", []):
                if ref.get("database_name") == "UniProt":
                    acc = ref.get("database_accession", "")
                    isoform = acc if "-" in acc else None
                    break

            reviewed = False
            if uniprot_id:
                _, reviewed = await uniprot_info(session, uniprot_id)

            return (
                uniprot_id,
                isoform or (f"{uniprot_id}-1" if uniprot_id else None),
                reviewed,
            )
    except Exception:
        pass
    return None, None, False


def pdb2uniprot_cached(
    session: aiohttp.ClientSession, pdb_id: str, chain_id: str
) -> asyncio.Task:
    """
    FIX 1 (PDB): Task-based cache — each (pdb_id, chain_id) fetched at most once,
    regardless of how many concurrent coroutines request it simultaneously.
    """
    key = (pdb_id, chain_id)
    if key not in _pdb_task_cache:
        _pdb_task_cache[key] = asyncio.ensure_future(
            _fetch_pdb2uniprot(session, pdb_id, chain_id)
        )
    return _pdb_task_cache[key]


# ── Target → UniProt dispatcher ────────────────────────────────────────────────

async def get_uniprot(
    session: aiohttp.ClientSession, target: str, database: str
) -> tuple:
    model_target = target.split(".")[0]

    if database == "alis_pdb100":
        pattern = r"^([0-9a-z]+)-(assembly[0-9]+)\.cif\.gz_([A-Za-z0-9]+)(?:-.*)?$"
        match = re.match(pattern, target)
        if not match:
            return _FALLBACK
        pdb_id, chain_id = match.group(1), match.group(3)
        uniprot, isoform, reviewed = await pdb2uniprot_cached(session, pdb_id, chain_id)
        return uniprot, chain_id, isoform, np.nan, np.nan, reviewed, False

    elif database == "alis_cath50":
        if model_target.startswith("af"):
            uniprot_id = model_target.split("_")[1]
            iso_num, reviewed = await uniprot_info(session, uniprot_id)
            return uniprot_id, np.nan, iso_num, np.nan, np.nan, reviewed, False
        else:
            pdb_id, chain_id = model_target[:4], model_target[4]
            uniprot, isoform, reviewed = await pdb2uniprot_cached(session, pdb_id, chain_id)
            return uniprot, chain_id, isoform, np.nan, np.nan, reviewed, False

    elif database == "alis_bfmd":
        if "ProtVar" in target:
            part = target.split("_")[3]
            A_parts = target.split("_")[1].split("-")
            B_parts = target.split("_")[2].split("-")

            A_uniprot = A_parts[0]
            if len(A_parts) > 1 and A_parts[1].isdigit():
                A_isoform = f"{A_parts[0]}-{A_parts[1]}"
                _, A_reviewed = await uniprot_info(session, A_uniprot)
            else:
                A_iso_num, A_reviewed = await uniprot_info(session, A_uniprot)
                A_isoform = f"{A_parts[0]}-{A_iso_num}"

            B_uniprot = B_parts[0]
            if len(B_parts) > 1 and B_parts[1].isdigit():
                B_isoform = f"{B_parts[0]}-{B_parts[1]}"
                _, B_reviewed = await uniprot_info(session, B_uniprot)
            else:
                B_iso_num, B_reviewed = await uniprot_info(session, B_uniprot)
                B_isoform = f"{B_parts[0]}-{B_iso_num}"

            if part == "A":
                return A_uniprot, np.nan, A_isoform, B_uniprot, B_isoform, A_reviewed, B_reviewed
            elif part == "B":
                return B_uniprot, np.nan, B_isoform, A_uniprot, A_isoform, B_reviewed, A_reviewed
            # part was neither A nor B — fall through to default below

    elif database in ("alis_afdb50", "alis_afdb-swissprot", "alis_afdb-proteome"):
        pattern = r"AF-(?P<uniprot>[A-Z0-9]+?)(?:-(?P<fragment>\d+))?-F1-model"
        match = re.match(pattern, model_target)
        if not match:
            return _FALLBACK
        uniprot_id = match.group("uniprot")
        isoform = f"{uniprot_id}-{match.group('fragment')}" if match.group("fragment") else np.nan
        _, reviewed = await uniprot_info(session, uniprot_id)
        return uniprot_id, np.nan, isoform, np.nan, np.nan, reviewed, False

    # ── FIX: Fallback for unhandled database types, unmatched patterns,
    #         alis_bfmd without ProtVar, or part not in {A, B} ──────────────
    return _FALLBACK


# ── Tar file reader ────────────────────────────────────────────────────────────

async def read_tartar_file(
    file_path: str,
    ensembl_id: str,
    session: aiohttp.ClientSession,
) -> pd.DataFrame:
    raw_dfs = []

    # Step 1: Read all CSVs from the archive first (fast synchronous disk I/O)
    with tarfile.open(file_path, "r:gz") as tar:
        for model in tar.getnames():
            if (
                "report" in model
                or "alis_mgnify_esm30" in model
                or "alis_gmgcl_id" in model
            ):
                continue
            try:
                df = (
                    pd.read_csv(tar.extractfile(model), sep="\t", header=None)
                    .iloc[:, list(range(1, 12)) + [15, 16] + list(range(-2, 0))]
                    .set_axis(columns, axis=1)
                )
                df[["target_model", "target_protein_name"]] = (
                    df["target_model"]
                    .str.split(" ", n=1, expand=True)
                    .reindex(columns=[0, 1])
                )
                df["database"] = model.split(".")[0]
                df["query_ensembl_id"] = ensembl_id
                raw_dfs.append(df)
            except pd.errors.EmptyDataError:
                continue
            except Exception:
                continue

    if not raw_dfs:
        return pd.DataFrame()

    # FIX 3: Build ALL HTTP tasks across ALL model files, then gather at once.
    # The original code gathered per-file sequentially; this processes everything
    # in a single concurrent batch.
    all_tasks = []
    lengths = []
    for df in raw_dfs:
        tasks = [
            get_uniprot(session, row["target_model"], row["database"])
            for _, row in df.iterrows()
        ]
        all_tasks.extend(tasks)
        lengths.append(len(df))

    all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Distribute results back to each df.
    # FIX: validate shape, not just exception type — guards against None returns
    # and any future wrong-length tuple from unhandled branches.
    offset = 0
    final_dfs = []
    for df, length in zip(raw_dfs, lengths):
        chunk = all_results[offset: offset + length]
        offset += length
        processed = [
            r if (isinstance(r, tuple) and len(r) == 7) else _FALLBACK
            for r in chunk
        ]
        results_df = pd.DataFrame(processed, index=df.index, columns=RESULT_COLS)

        # Drop rows where get_uniprot returned the fallback (all first 5 result
        # columns are null). The fallback tuple uses NaN for the first five
        # fields, so if all are null we treat it as an unresolved hit and omit
        # the row entirely.
        first_five = RESULT_COLS[:5]
        keep_mask = ~results_df[first_five].isnull().all(axis=1)
        if keep_mask.any():
            df = df.loc[keep_mask].copy()
            df[RESULT_COLS] = results_df.loc[df.index]
            final_dfs.append(df)
        # if no rows remain after filtering, skip adding this dataframe

    return pd.concat(final_dfs, ignore_index=True)


# ── Query metadata enrichment ──────────────────────────────────────────────────

async def enrich_with_query_info(
    df: pd.DataFrame,
    ensembl_id: str,
    session: aiohttp.ClientSession,
) -> pd.DataFrame:
    """
    FIX 5: Uses pre-indexed query_info_indexed for O(1) lookup.
    The original did a full DataFrame scan (O(n)) on every single file.
    """
    row = (
        query_info_indexed.loc[[ensembl_id]]
        if ensembl_id in query_info_indexed.index
        else pd.DataFrame()
    )

    def _get(col):
        if row.empty or col not in row.columns:
            return np.nan
        val = row[col].values[0]
        return np.nan if pd.isna(val) else val

    df["query_gene_name"]           = _get("gene_name")
    df["query_uniprot_id"]          = _get("uniprot_id")
    df["query_isoform"]             = _get("isoform")
    df["query_uniprot_id_sequence"] = _get("sequence")

    reviewed_val = _get("reviewed")
    if not pd.isna(reviewed_val):
        df["query_uniprot_reviewed"] = bool(reviewed_val)
    else:
        qid = _get("uniprot_id")
        if not pd.isna(qid):
            _, qreviewed = await uniprot_info(session, str(qid))
            df["query_uniprot_reviewed"] = qreviewed
        else:
            df["query_uniprot_reviewed"] = False

    return df


# ── Per-Ensembl-ID unit ────────────────────────────────────────────────────────

async def process_ensembl_id(
    ensembl_id: str,
    files: list,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    async with semaphore:
        ensembl_dfs = []
        for file_path, chain in files:
            df = await read_tartar_file(file_path, ensembl_id, session)
            if not df.empty:
                df["query_chain_id"] = chain
                df = await enrich_with_query_info(df, ensembl_id, session)
                ensembl_dfs.append(df)

        if ensembl_dfs:
            return pd.concat(ensembl_dfs, ignore_index=True)
        return pd.DataFrame()


# ── Main orchestrator ──────────────────────────────────────────────────────────

async def process_all_files():
    output_file = "foldseek_combined_results_new.pkl"

    if os.path.exists(output_file):
        existing_df = pd.read_pickle(output_file)
        processed_ids = set(
            existing_df[["query_ensembl_id", "query_chain_id"]].apply(tuple, axis=1)
        )
        # Pre-populate the UniProt cache from already-fetched data so we don't
        # re-hit the API for IDs we've already resolved in a previous run.
        for col, rev_col in [
            ("target_uniprot_id", "target_uniprot_reviewed"),
            ("target_dimer_uniprot_id", "target_dimer_uniprot_reviewed"),
        ]:
            if col in existing_df.columns and rev_col in existing_df.columns:
                for uid, reviewed in (
                    existing_df[[col, rev_col]]
                    .dropna(subset=[col])
                    .drop_duplicates(subset=[col])
                    .itertuples(index=False)
                ):
                    base = str(uid).split("-")[0]
                    if base not in _uniprot_task_cache:
                        iso_num = "1"
                        async def _cached(i=iso_num, r=bool(reviewed)):
                            return i, r
                        _uniprot_task_cache[base] = asyncio.ensure_future(_cached())
    else:
        existing_df = pd.DataFrame()
        processed_ids = set()

    # Build file map grouped by Ensembl ID
    ensembl_files: dict = {}
    for result in os.listdir("foldseek_results"):
        if result.endswith(".tar.gz"):
            ensembl_id = result.replace(".tar.gz", "")
            if (ensembl_id, None) not in processed_ids:
                ensembl_files.setdefault(ensembl_id, []).append(
                    [f"foldseek_results/{result}", None]
                )
        else:
            ensembl_id = result
            for chain in os.listdir(f"foldseek_results/{result}"):
                chain_id = chain.split(".")[0]
                if (ensembl_id, chain_id) not in processed_ids:
                    ensembl_files.setdefault(ensembl_id, []).append(
                        [f"foldseek_results/{result}/{chain}", chain_id]
                    )

    ensembl_id_list = sorted(ensembl_files.keys())
    semaphore = asyncio.Semaphore(ENSEMBL_CONCURRENCY)

    # FIX 2: One shared ClientSession for the entire run.
    # limit_per_host=30 avoids hammering UniProt/RCSB with too many parallel connections.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=200,
            limit_per_host=30,
            enable_cleanup_closed=True,
        ),
        timeout=aiohttp.ClientTimeout(total=60),
    ) as session:

        # FIX 6: All Ensembl IDs submitted as Tasks immediately;
        # the semaphore limits how many actually run concurrently.
        # Awaited in sorted order for an accurate tqdm progress bar
        # while still benefiting from concurrent execution.
        id_to_task = {
            eid: asyncio.ensure_future(
                process_ensembl_id(eid, ensembl_files[eid], session, semaphore)
            )
            for eid in ensembl_id_list
        }

        for ensembl_id in tqdm(ensembl_id_list, desc="Processing Ensembl IDs"):
            try:
                combined = await id_to_task[ensembl_id]
                if not combined.empty:
                    if existing_df.empty:
                        existing_df = combined
                    else:
                        existing_df = pd.concat([existing_df, combined], ignore_index=True)
                    existing_df.to_pickle(output_file)
            except Exception as e:
                print(f"[ERROR] {ensembl_id}: {e}")


asyncio.run(process_all_files())
