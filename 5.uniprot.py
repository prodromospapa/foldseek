# stdlib
import gc
import gzip
import json
import os
import pickle
import re
import runpy
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import deque

# third-party
import pandas as pd
import wget
from goatools.obo_parser import GODag
from tqdm import tqdm
import xml.etree.ElementTree as etree

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# ── Constants ─────────────────────────────────────────────────────────────────

UNIPROT_DB_FILE = 'uniprot_index.db'
RESULTS_FILE = 'foldseek_combined_results.pkl'
ANNOTATED_FILE = 'foldseek_combined_results_with_info.pkl'
PROGRESS_FILE = 'processed_uniprot_ids.pkl'
GO_HIERARCHY_FILE = 'go_hierarchy.pkl'

# Memory thresholds (as fraction of total RAM)
FLUSH_RAM_PCT         = 0.20   # Flush to DB when RAM usage hits this
MEMORY_HARD_LIMIT_PCT = 0.40   # Crash if usage exceeds this

# SQLite configuration
SQLITE_PARAM_LIMIT  = 900
PRAGMA_CACHE_SIZE   = -1000000
PRAGMA_JOURNAL_MODE = 'WAL'
PRAGMA_SYNCHRONOUS  = 'OFF'

UNIPROT_SOURCES = [
    ('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz',
     'uniprot_sprot.xml.gz', 1),  # reviewed (Swiss-Prot)
    ('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.xml.gz',
     'uniprot_trembl.xml.gz', 0),  # unreviewed (TrEMBL)
]


def _get_total_memory_gb():
    if HAS_PSUTIL and psutil is not None:
        return psutil.virtual_memory().total / (1024 ** 3)
    return None


def _count_xml_entries(gz_file):
    """Count <entry occurrences in a gzip XML file using zcat|grep -o|wc -l.
    Returns None if unavailable so tqdm runs without a total."""
    try:
        result = subprocess.run(
            ['bash', '-c', f'zcat {gz_file} | grep -o "<entry " | wc -l'],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except Exception:
        return None


def _fetch_online_uniprot_totals():
    """Fetch current UniProtKB totals for reviewed/unreviewed entries from UniProt REST.

    Returns dict: {1: swissprot_total_or_None, 0: trembl_total_or_None}
    where key 1 = reviewed (Swiss-Prot), 0 = unreviewed (TrEMBL).
    """
    totals: dict[int, int | None] = {1: None, 0: None}
    queries = {
        1: 'reviewed:true',
        0: 'reviewed:false',
    }

    for reviewed_flag, query in queries.items():
        try:
            encoded_q = urllib.parse.quote(query, safe='')
            url = (
                'https://rest.uniprot.org/uniprotkb/search'
                f'?query={encoded_q}&format=json&size=1&fields=accession'
            )
            req = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                header_total = resp.headers.get('X-Total-Results')
                if header_total and header_total.isdigit():
                    totals[reviewed_flag] = int(header_total)
                    continue

                payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
                if isinstance(payload, dict):
                    fallback_total = payload.get('totalResults')
                    if isinstance(fallback_total, int):
                        totals[reviewed_flag] = fallback_total
        except Exception:
            totals[reviewed_flag] = None

    return totals


def normalize_term(value):
    """Normalize ontology terms for consistent comparisons."""
    return (value or '').strip().rstrip('.').lower()


_TOTAL_MEM_GB        = _get_total_memory_gb()
MEMORY_HARD_LIMIT_GB = (_TOTAL_MEM_GB * MEMORY_HARD_LIMIT_PCT) if _TOTAL_MEM_GB else 85.0
FLUSH_RAM_LIMIT_GB   = (_TOTAL_MEM_GB * FLUSH_RAM_PCT)         if _TOTAL_MEM_GB else 16.0
BATCH_RAM_LIMIT_GB   = FLUSH_RAM_LIMIT_GB


def get_memory_usage_gb():
    """Get total system RAM currently in use, in GB."""
    if HAS_PSUTIL and psutil is not None:
        return psutil.virtual_memory().used / (1024 ** 3)

    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            meminfo = dict(line.split(':', 1) for line in f)
        total_kb = int(meminfo['MemTotal'].split()[0])
        avail_kb = int(meminfo['MemAvailable'].split()[0])
        return (total_kb - avail_kb) / (1024 ** 2)
    except (OSError, KeyError, ValueError):
        return None


def check_memory_safety(check_name='operation'):
    """Raise MemoryError if RAM usage exceeds the hard limit."""
    mem_used_gb = get_memory_usage_gb()
    if mem_used_gb is None:
        return
    if mem_used_gb >= MEMORY_HARD_LIMIT_GB:
        raise MemoryError(
            f'Memory limit exceeded at {check_name}: '
            f'{mem_used_gb:.1f}GB used > {MEMORY_HARD_LIMIT_GB:.1f}GB '
            f'({MEMORY_HARD_LIMIT_PCT*100:.0f}% of {_TOTAL_MEM_GB:.1f}GB total)'
        )


def safe_collect():
    """Force garbage collection and check memory."""
    gc.collect()
    check_memory_safety('gc')


def load_go_dict():
    """Load GO ID -> UniProt subcellular-location ID mapping from prebuilt cache."""
    if not os.path.exists('go_dict.pkl'):
        raise FileNotFoundError('go_dict.pkl not found. Run 5.subcell_hierarchy_uniprot.py first.')
    with open('go_dict.pkl', 'rb') as f:
        return pickle.load(f)


def load_go_hierarchy():
    """Load GO ID -> GO ancestor list mapping from prebuilt cache."""
    if not os.path.exists(GO_HIERARCHY_FILE):
        raise FileNotFoundError(f'{GO_HIERARCHY_FILE} not found. Run 5.subcell_hierarchy_uniprot.py first.')
    with open(GO_HIERARCHY_FILE, 'rb') as f:
        return pickle.load(f)


def ensure_subcell_assets():
    """Ensure required subcell ontology assets exist; build them via 5.1 if needed."""
    required = ['subcell.txt', 'go-basic.obo', 'subcell_hierarchy.pkl', 'go_dict.pkl', GO_HIERARCHY_FILE]
    missing = [p for p in required if not os.path.exists(p)]
    if not missing:
        return

    print(f"Missing ontology assets: {', '.join(missing)}", file=sys.stderr)
    print('Running 5.subcell_hierarchy_uniprot.py to fetch/build missing assets...', file=sys.stderr)
    script_path = os.path.join(os.path.dirname(__file__), '5.subcell_hierarchy_uniprot.py')
    if not os.path.exists(script_path):
        raise FileNotFoundError(f'Missing builder script: {script_path}')
    runpy.run_path(script_path, run_name='__main__')

    still_missing = [p for p in required if not os.path.exists(p)]
    if still_missing:
        raise FileNotFoundError(
            'Missing required ontology assets after running 5.1: '
            + ', '.join(still_missing)
        )


def _expand_subcell_terms(locations, uniprot_dict):
    """Expand UniProt subcellular-location terms through subcell_hierarchy.pkl."""
    expanded = []
    seen = set()
    for loc in locations:
        loc_l = (loc or '').strip().lower()
        if not loc_l:
            continue
        if loc_l in uniprot_dict:
            for anc in uniprot_dict[loc_l]:
                anc_l = anc.lower()
                if anc_l not in seen:
                    seen.add(anc_l)
                    expanded.append(anc_l)
        elif loc_l not in seen:
            seen.add(loc_l)
            expanded.append(loc_l)
    return expanded


# GO expansion cache to avoid redundant traversals
_GO_EXPANSION_CACHE = {}

def _expand_go_annotations(go_entries, go_hierarchy, godag=None):
    """Expand GO cellular-component annotations through the prebuilt GO hierarchy."""
    global _GO_EXPANSION_CACHE
    expanded = []
    seen = set()

    for go in go_entries:
        go_id = go.get('go_id')
        if not go_id or go_id in seen:
            continue

        # Check cache first
        if go_id not in _GO_EXPANSION_CACHE:
            cached_expansion = []
            ancestor_entries = go_hierarchy.get(go_id, [{'go_id': go_id, 'term': go_id}])
            for entry in ancestor_entries:
                if isinstance(entry, dict):
                    curr_id = entry.get('go_id', '')
                    term_name = entry.get('term') or entry.get('name') or curr_id
                else:
                    curr_id = entry
                    term_name = entry

                curr_id = (curr_id or '').strip()
                term_name = (term_name or '').strip()

                if not curr_id:
                    continue
                if not term_name and godag is not None:
                    curr_term = godag.get(curr_id)
                    term_name = normalize_term(curr_term.name) if curr_term and curr_term.name else curr_id
                elif not term_name:
                    term_name = curr_id

                cached_expansion.append({'go_id': curr_id, 'term': term_name})

            _GO_EXPANSION_CACHE[go_id] = cached_expansion

        # Use cached result
        for item in _GO_EXPANSION_CACHE[go_id]:
            if item['go_id'] not in seen:
                seen.add(item['go_id'])
                expanded.append(item)

    return expanded


def _merge_localization_ancestors(uniprot_terms, go_terms):
    """Merge UniProt and GO ancestor annotations into one deduplicated list."""
    merged = []
    seen = set()

    for term in list(uniprot_terms or []) + list(go_terms or []):
        normalized = normalize_term(term)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)

    return merged


def _tag_endswith(el, name):
    """Check if element tag ends with name."""
    return getattr(el, 'tag', '').endswith('}' + name) or el.tag == name


def _extract_pos(el):
    """Extract an integer position from a UniProt XML position element."""
    if el is None:
        return None

    p = el.get('position')
    if p and p.isdigit():
        return int(p)

    for c in el:
        if _tag_endswith(c, 'position') and c.text and c.text.strip().isdigit():
            return int(c.text.strip())

    if el.text and el.text.strip().isdigit():
        return int(el.text.strip())
    return None


def parse_entry_xml(root, go_dict, uniprot_dict, go_hierarchy, godag):
    """Parse a UniProt XML <entry> element and return a dict of annotation fields."""
    result = {
        'ensembl_id': '',
        'gene_name': '',
        'protein_name': '',
        'organism': '',
        'localization_uniprot': [],
        'localization_go': [],
        'domains': [],
        'panthr_id': [],
    }

    seen_go_terms = set()
    domain_types = {'domain', 'repeat', 'motif', 'zinc finger', 'region'}

    # Build parent map once for O(1) lookups (Issue 1.1 optimization)
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[id(child)] = parent

    # Pre-index elements by tag name for fast lookup (Issue 1.2 optimization)
    has_protein = has_gene = has_ensembl = has_organism = False

    # Single pass through element tree with parent map
    for el in root.iter():
        el_id = id(el)
        el_parent = parent_map.get(el_id)

        # Extract tag name once (Issue 1.3 optimization - cache tag names)
        tag = el.tag
        tag_local = tag.split('}')[-1] if '}' in tag else tag

        # Protein name
        if tag_local == 'fullName' and not has_protein and el.text:
            if el_parent is not None:
                parent_tag = el_parent.tag
                parent_local = parent_tag.split('}')[-1] if '}' in parent_tag else parent_tag
                if parent_local == 'fullName' or parent_local == 'recommendedName' or parent_local == 'submittedName':
                    result['protein_name'] = el.text.strip()
                    has_protein = True

        # Gene name
        elif tag_local == 'name' and el.text:
            if el_parent is not None:
                parent_tag = el_parent.tag
                parent_local = parent_tag.split('}')[-1] if '}' in parent_tag else parent_tag
                if parent_local == 'gene' and not has_gene:
                    result['gene_name'] = el.text.strip()
                    has_gene = True
                elif parent_local == 'organism' and not has_organism and el.get('type') == 'scientific':
                    result['organism'] = el.text.strip().lower()
                    has_organism = True

        # Database references
        elif tag_local == 'dbReference':
            dbtype = el.get('type')
            dbid = el.get('id')
            if dbtype and dbid:
                if dbtype == 'Ensembl' and not has_ensembl:
                    result['ensembl_id'] = dbid.split('.')[0]
                    has_ensembl = True
                elif dbtype == 'PANTHER':
                    result['panthr_id'].append(dbid)
                elif dbtype == 'GO':
                    for prop in el:
                        prop_tag = prop.tag.split('}')[-1] if '}' in prop.tag else prop.tag
                        if prop_tag == 'property':
                            val = prop.get('value') or (prop.text.strip() if prop.text else '')
                            if val.startswith('C:'):
                                term = val[2:]
                                if term not in seen_go_terms:
                                    seen_go_terms.add(term)
                                    result['localization_go'].append({'go_id': dbid, 'term': term})

        # Subcellular locations
        elif tag_local == 'location' and el_parent is not None:
            parent_tag = el_parent.tag
            parent_local = parent_tag.split('}')[-1] if '}' in parent_tag else parent_tag
            if parent_local in ('subcellularLocation', 'subcellularLocations'):
                v = next((c for c in el if (c.tag.split('}')[-1] if '}' in c.tag else c.tag) == 'value'), None)
                val = (v.text.strip() if v is not None and v.text else None) or \
                      (el.text.strip() if el.text else None)
                if val and val not in result['localization_uniprot']:
                    result['localization_uniprot'].append(val)

        # Features (domains, etc.)
        elif tag_local == 'feature':
            ftype = el.get('type', '').lower()
            if ftype not in domain_types:
                continue

            loc = next((c for c in el if (c.tag.split('}')[-1] if '}' in c.tag else c.tag) == 'location'), None)
            if loc is not None:
                # Extract all positions at once (Issue 7.1 optimization)
                begin = _extract_pos(next((c for c in loc if (c.tag.split('}')[-1] if '}' in c.tag else c.tag) == 'begin'), None))
                end = _extract_pos(next((c for c in loc if (c.tag.split('}')[-1] if '}' in c.tag else c.tag) == 'end'), None))
                position = _extract_pos(next((c for c in loc if (c.tag.split('}')[-1] if '}' in c.tag else c.tag) == 'position'), None))

                start = begin if begin is not None else position
                end_val = end if end is not None else position

                if start is not None and end_val is not None:
                    result['domains'].append({
                        'type': el.get('type', ''),
                        'description': el.get('description', ''),
                        'start': start,
                        'end': end_val,
                    })

    result['localization_uniprot'] = _expand_subcell_terms(result['localization_uniprot'], uniprot_dict)
    result['localization_go'] = _expand_go_annotations(result['localization_go'], go_hierarchy, godag)
    result['localization_ancestors'] = _merge_localization_ancestors(
        result['localization_uniprot'],
        [item.get('term', '') for item in result['localization_go']],
    )

    return result


def setup_uniprot_database(go_dict, uniprot_dict, go_hierarchy, godag, db_file=UNIPROT_DB_FILE):
    """Download and index Swiss-Prot + TrEMBL into SQLite."""
    db_exists = os.path.exists(db_file)
    conn = sqlite3.connect(db_file)
    conn.execute(f'PRAGMA journal_mode = {PRAGMA_JOURNAL_MODE}')
    conn.execute(f'PRAGMA synchronous = {PRAGMA_SYNCHRONOUS}')
    conn.execute(f'PRAGMA cache_size = {PRAGMA_CACHE_SIZE}')
    conn.execute('PRAGMA temp_store = MEMORY')
    conn.execute('CREATE TABLE IF NOT EXISTS entries '
                 '(accession TEXT PRIMARY KEY, reviewed INTEGER NOT NULL, organism TEXT NOT NULL, data BLOB NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS source_index_log '
                 '(source TEXT PRIMARY KEY, completed INTEGER NOT NULL DEFAULT 0, indexed_entries INTEGER NOT NULL DEFAULT 0)')
    conn.commit()

    row_count = conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
    if db_exists:
        if row_count > 0:
            print(f'Using existing UniProt database: {db_file} ({row_count:,} rows)', file=sys.stderr)
        else:
            print(f'Existing database found but empty ({db_file}); rebuilding index...', file=sys.stderr)
    else:
        print(f'No existing UniProt database found at {db_file}; building index...', file=sys.stderr)

    # Backfill source-level progress for legacy DBs that predate source_index_log.
    existing_log_rows = conn.execute('SELECT COUNT(*) FROM source_index_log').fetchone()[0]
    if existing_log_rows == 0 and row_count > 0:
        reviewed_count = conn.execute('SELECT COUNT(*) FROM entries WHERE reviewed = 1').fetchone()[0]
        unreviewed_count = conn.execute('SELECT COUNT(*) FROM entries WHERE reviewed = 0').fetchone()[0]

        if reviewed_count > 0:
            conn.execute(
                'INSERT OR REPLACE INTO source_index_log(source, completed, indexed_entries) VALUES (?,?,?)',
                ('uniprot_sprot.xml.gz', 1, reviewed_count),
            )
        if unreviewed_count > 0:
            conn.execute(
                'INSERT OR REPLACE INTO source_index_log(source, completed, indexed_entries) VALUES (?,?,?)',
                ('uniprot_trembl.xml.gz', 1, unreviewed_count),
            )
        conn.commit()

        if reviewed_count > 0 or unreviewed_count > 0:
            print('Bootstrapped source_index_log from existing reviewed/unreviewed rows.', file=sys.stderr)

    source_status = {}
    for _, xml_file, _ in UNIPROT_SOURCES:
        row = conn.execute(
            'SELECT completed FROM source_index_log WHERE source = ?', (xml_file,)
        ).fetchone()
        source_status[xml_file] = bool(row and row[0])

    if all(source_status.values()):
        conn.execute('CREATE INDEX IF NOT EXISTS idx_accession ON entries(accession)')
        conn.commit()
        conn.close()
        print('Swiss-Prot and TrEMBL already indexed; skipping UniProt indexing.', file=sys.stderr)
        return

    upsert_sql = (
        'INSERT INTO entries(accession, reviewed, organism, data) VALUES (?,?,?,?) '
        'ON CONFLICT(accession) DO UPDATE SET '
        'reviewed = CASE WHEN excluded.reviewed > entries.reviewed THEN excluded.reviewed ELSE entries.reviewed END, '
        'organism = CASE WHEN excluded.reviewed > entries.reviewed THEN excluded.organism ELSE entries.organism END, '
        'data = CASE WHEN excluded.reviewed > entries.reviewed THEN excluded.data ELSE entries.data END'
    )

    for url, xml_file, reviewed_flag in UNIPROT_SOURCES:
        if source_status.get(xml_file, False):
            print(f'\nSkipping {xml_file}: already indexed (source_index_log).', file=sys.stderr)
            continue

        use_local = os.path.exists(xml_file)
        online_totals = _fetch_online_uniprot_totals()
        total_entries = online_totals.get(reviewed_flag)
        source_name = 'Swiss-Prot' if reviewed_flag == 1 else 'TrEMBL'
        if total_entries is not None:
            print(f'\nRefreshed online total for {source_name}: {total_entries:,} entries', file=sys.stderr)
        else:
            print(f'\nOnline total unavailable for {source_name}; falling back to local counting if possible.', file=sys.stderr)

        if use_local:
            if total_entries is not None:
                print(f'\nUsing online total for {xml_file}: {total_entries:,} entries', file=sys.stderr)
            else:
                print(f'\nCounting entries in {xml_file}...', file=sys.stderr)
                sys.stderr.flush()
                total_entries = _count_xml_entries(xml_file)
                if total_entries is not None:
                    print(f'  {total_entries:,} entries found', file=sys.stderr)
                else:
                    print(f'  Count unavailable, progress bar will show absolute numbers', file=sys.stderr)
        else:
            print(f'\nStreaming {xml_file} from {url} (no local copy to save disk space)...', file=sys.stderr)
            if total_entries is not None:
                print(f'  Online total: {total_entries:,} entries', file=sys.stderr)
            else:
                print('  Online total unavailable, progress bar will show absolute numbers', file=sys.stderr)

        print(f'Indexing {xml_file}...', file=sys.stderr)
        sys.stderr.flush()

        def _open_source():
            if use_local:
                return gzip.open(xml_file, 'rt', encoding='utf-8', errors='ignore')
            else:
                response = urllib.request.urlopen(url)
                return gzip.open(response, 'rt', encoding='utf-8', errors='ignore')

        n, batch = 0, []
        source_complete = True
        try:
            with _open_source() as f:
                pbar = tqdm(total=total_entries, unit='entry', desc=xml_file, file=sys.stderr)
                for event, elem in etree.iterparse(f, events=('end',)):
                    if not (elem.tag.endswith('}entry') or elem.tag == 'entry'):
                        continue

                    try:
                        accs = [acc.text for acc in elem.iter() if
                               (acc.tag.endswith('}accession') or acc.tag == 'accession') and acc.text]

                        if accs:
                            parsed = parse_entry_xml(elem, go_dict, uniprot_dict, go_hierarchy, godag)
                            data_blob = pickle.dumps(parsed, protocol=5)
                            organism = parsed.get('organism', '')
                            batch.extend((acc, reviewed_flag, organism, data_blob) for acc in accs)
                            n += 1
                            pbar.update(1)

                            if n % 10000 == 0:
                                mem_gb = ram_used_gb()
                                pbar.set_postfix_str(f'{mem_gb:.1f}/{_TOTAL_MEM_GB:.0f}GB RAM')
                                check_memory_safety('indexing')
                                if mem_gb is not None and mem_gb >= FLUSH_RAM_LIMIT_GB:
                                    conn.executemany(upsert_sql, batch)
                                    conn.commit()
                                    batch = []
                                    gc.collect()
                    except Exception:
                        pass
                    finally:
                        elem.clear()
                pbar.close()

        except EOFError:
            print(f'\nGzip truncated for {xml_file}; partial data indexed.', file=sys.stderr)
            source_complete = False
        finally:
            safe_collect()

        if batch:
            conn.executemany(upsert_sql, batch)
            conn.commit()
        print(f'\n  ✓ Done: {n:,} entries from {xml_file}', file=sys.stderr)
        sys.stderr.flush()
        safe_collect()

        if source_complete:
            conn.execute(
                'INSERT OR REPLACE INTO source_index_log(source, completed, indexed_entries) VALUES (?,?,?)',
                (xml_file, 1, n),
            )
            conn.commit()
        else:
            print(f'  Not marking {xml_file} complete due to truncated stream; it will resume on rerun.', file=sys.stderr)

        if use_local and os.path.exists(xml_file):
            os.remove(xml_file)
            print(f'  Deleted {xml_file}', file=sys.stderr)

    conn.execute('CREATE INDEX IF NOT EXISTS idx_accession ON entries(accession)')
    conn.commit()

    conn.close()
    print('UniProt database created.', file=sys.stderr)


def ram_used_gb():
    """Current RAM usage in GB, or None if unavailable."""
    return get_memory_usage_gb()


def fetch_entries_batch(accessions, db_file=UNIPROT_DB_FILE):
    """Fetch accessions from SQLite, stopping each batch when RAM hits BATCH_RAM_LIMIT_GB.

    Yields (batch_cache dict, remaining_accessions list) so the caller can process
    and checkpoint before the next batch is loaded.
    """
    accessions = list(accessions)
    if not accessions:
        return

    conn = sqlite3.connect(db_file)
    batch = {}
    i = 0

    while i < len(accessions):
        # Check RAM before each SQLite chunk
        mem = ram_used_gb()
        if mem is not None and mem >= BATCH_RAM_LIMIT_GB:
            # Yield what we have so far, caller annotates and frees memory
            yield batch, accessions[i:]
            batch = {}
            gc.collect()
            continue

        chunk = accessions[i:i + SQLITE_PARAM_LIMIT]  # SQLite hard limit: max 999 params per query
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT accession, reviewed, organism, data FROM entries WHERE accession IN ({placeholders})',
            chunk,
        ).fetchall()

        for acc, reviewed, organism, data in rows:
            batch[acc] = {**pickle.loads(data), 'reviewed': bool(reviewed), 'organism': organism}

        i += len(chunk)

    conn.close()
    if batch:
        yield batch, []


def atomic_pickle_dump(obj, file_path):
    """Write a pickle atomically to avoid corruption on interruption."""
    tmp = f'{file_path}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, file_path)


def _assign_object_column(frame, indices, column, values):
    """Assign values to a DataFrame column using object dtype."""
    if len(indices) > 0:
        frame.loc[indices, column] = pd.Series(values, index=indices, dtype='object')


def _assign_list_column(frame, indices, column, value):
    """Assign copies of a list to multiple rows."""
    _assign_object_column(frame, indices, column, [value.copy() for _ in range(len(indices))])


def _extend_domains(existing_lists, starts, ends, domain_spans):
    """Merge overlapping domain descriptions into existing domain lists."""
    updated = []
    for existing, start, end in zip(existing_lists, starts, ends):
        merged = existing.copy() if isinstance(existing, list) else []
        merged.extend(
            desc for d_start, d_end, desc in domain_spans
            if d_start <= end and d_end >= start
        )
        updated.append(merged)
    return updated


def load_results():
    """Load the annotated results DataFrame, rebuilding from base if needed."""
    if os.path.exists(ANNOTATED_FILE):
        try:
            df = pd.read_pickle(ANNOTATED_FILE)
            base_df = pd.read_pickle(RESULTS_FILE)
            if 'target_reviewed' not in df.columns or len(df) != len(base_df):
                print('Detected legacy results file; rebuilding from base results.', file=sys.stderr)
                df = base_df.copy()
            del base_df
        except (EOFError, pickle.UnpicklingError):
            print('Corrupted annotated results file; rebuilding.', file=sys.stderr)
            df = pd.read_pickle(RESULTS_FILE)
    else:
        df = pd.read_pickle(RESULTS_FILE)

    # Remove obsolete columns
    if 'target_uniprot_id_sequence' in df.columns:
        df = df.drop(columns=['target_uniprot_id_sequence'])

    return df


def ensure_columns(df):
    """Add missing annotation columns. Returns True if any were added."""
    added = False

    list_cols = [
        'query_domains', 'query_panthr_id', 'query_localization_uniprot',
        'query_localization_ancestors', 'query_localization_go',
        'target_domains', 'target_panthr_id', 'target_localization_uniprot',
        'target_localization_ancestors', 'target_localization_go',
        'target_dimer_domains', 'target_dimer_panthr_id', 'target_dimer_localization_uniprot',
        'target_dimer_localization_ancestors', 'target_dimer_localization_go',
    ]
    for col in list_cols:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
            added = True

    scalar_cols = {
        'query_protein_name': '',
        'query_reviewed': False,
        'target_gene_name': '',
        'target_ensembl_id': '',
        'target_reviewed': False,
        'target_dimer_protein_name': '',
    }
    for col, default in scalar_cols.items():
        if col not in df.columns:
            df[col] = default
            added = True

    return added


def load_progress():
    """Return the list of already-processed UniProt IDs from checkpoint file."""
    if not os.path.exists(PROGRESS_FILE):
        print(f'No progress checkpoint found ({PROGRESS_FILE}); starting from 0 processed IDs.', file=sys.stderr)
        return []
    try:
        return pd.read_pickle(PROGRESS_FILE)['uniprot_id'].tolist()
    except (EOFError, pickle.UnpicklingError):
        print('Corrupted progress file; restarting progress tracking.', file=sys.stderr)
        return []


def _normalize_uniprot_ids(values):
    """Return cleaned UniProt IDs as non-empty strings with NaN removed."""
    cleaned = []
    for val in values:
        if pd.isna(val):
            continue
        s = val.strip() if isinstance(val, str) else str(val).strip()
        if s and s.lower() != 'nan':
            cleaned.append(s)
    return cleaned


def annotate_id(df, uniprot_id, info, query_map, target_map, dimer_map):
    """Write UniProt annotation for one accession into matching rows of df."""
    if not info:
        return

    query_indices = query_map.get(uniprot_id, [])
    target_indices = target_map.get(uniprot_id, [])
    dimer_indices = dimer_map.get(uniprot_id, [])

    # Extract info once
    panthr_list = info.get('panthr_id', [])
    loc_uniprot = info.get('localization_uniprot', [])
    loc_ancestors = info.get('localization_ancestors', [])
    loc_go = info.get('localization_go', [])
    reviewed = info.get('reviewed', False)
    domain_spans = [
        (d['start'], d['end'], d['description'])
        for d in info.get('domains', [])
        if d.get('description') and d.get('start') is not None and d.get('end') is not None
    ]

    # Annotate query
    if len(query_indices) > 0:
        df.loc[query_indices, 'query_protein_name'] = info.get('protein_name', '')
        df.loc[query_indices, 'query_reviewed'] = reviewed
        _assign_list_column(df, query_indices, 'query_panthr_id', panthr_list)
        _assign_list_column(df, query_indices, 'query_localization_uniprot', loc_uniprot)
        _assign_list_column(df, query_indices, 'query_localization_ancestors', loc_ancestors)
        _assign_list_column(df, query_indices, 'query_localization_go', loc_go)
        if domain_spans:
            qstarts = df.loc[query_indices, 'qstart'].to_numpy(copy=False)
            qends = df.loc[query_indices, 'qend'].to_numpy(copy=False)
            _assign_object_column(df, query_indices, 'query_domains',
                _extend_domains(df.loc[query_indices, 'query_domains'].tolist(), qstarts, qends, domain_spans))

    # Annotate target
    if len(target_indices) > 0:
        df.loc[target_indices, 'target_gene_name'] = info.get('gene_name', '')
        df.loc[target_indices, 'target_ensembl_id'] = info.get('ensembl_id', '')
        df.loc[target_indices, 'target_reviewed'] = reviewed
        _assign_list_column(df, target_indices, 'target_panthr_id', panthr_list)
        _assign_list_column(df, target_indices, 'target_localization_uniprot', loc_uniprot)
        _assign_list_column(df, target_indices, 'target_localization_ancestors', loc_ancestors)
        _assign_list_column(df, target_indices, 'target_localization_go', loc_go)
        if domain_spans:
            tstarts = df.loc[target_indices, 'tstart'].to_numpy(copy=False)
            tends = df.loc[target_indices, 'tend'].to_numpy(copy=False)
            _assign_object_column(df, target_indices, 'target_domains',
                _extend_domains(df.loc[target_indices, 'target_domains'].tolist(), tstarts, tends, domain_spans))

    # Annotate dimer
    if len(dimer_indices) > 0:
        df.loc[dimer_indices, 'target_dimer_protein_name'] = info.get('protein_name', '')
        _assign_list_column(df, dimer_indices, 'target_dimer_panthr_id', panthr_list)
        _assign_list_column(df, dimer_indices, 'target_dimer_localization_uniprot', loc_uniprot)
        _assign_list_column(df, dimer_indices, 'target_dimer_localization_ancestors', loc_ancestors)
        _assign_list_column(df, dimer_indices, 'target_dimer_localization_go', loc_go)


def main():
    print('Loading ontology resources...', file=sys.stderr)
    sys.stderr.flush()

    ensure_subcell_assets()

    uniprot_dict = pd.read_pickle('subcell_hierarchy.pkl')
    print(f'  ✓ Subcellular terms: {len(uniprot_dict)}', file=sys.stderr)
    sys.stderr.flush()

    print('  Loading GO hierarchy cache...', file=sys.stderr)
    sys.stderr.flush()
    go_hierarchy = load_go_hierarchy()
    print(f'  ✓ GO hierarchy terms: {len(go_hierarchy)}', file=sys.stderr)
    sys.stderr.flush()

    print('  Loading GO ontology (may take 10-30 seconds)...', file=sys.stderr)
    sys.stderr.flush()
    godag = GODag('go-basic.obo', prt=None)
    print(f'  ✓ GO terms: {len(godag)}', file=sys.stderr)
    sys.stderr.flush()

    print('  Loading GO→UniProt mappings...', file=sys.stderr)
    sys.stderr.flush()
    go_dict = load_go_dict()
    print(f'  ✓ GO→UniProt mappings: {len(go_dict)}', file=sys.stderr)
    sys.stderr.flush()

    print(f'\nSystem: {_TOTAL_MEM_GB:.1f}GB RAM | flush limit {FLUSH_RAM_PCT*100:.0f}% ({FLUSH_RAM_LIMIT_GB:.1f}GB) | hard limit {MEMORY_HARD_LIMIT_PCT*100:.0f}% ({MEMORY_HARD_LIMIT_GB:.1f}GB)', file=sys.stderr)
    sys.stderr.flush()

    print('\nSetting up UniProt database...', file=sys.stderr)
    sys.stderr.flush()
    setup_uniprot_database(go_dict, uniprot_dict, go_hierarchy, godag)

    print('Loading results...', file=sys.stderr)
    sys.stderr.flush()
    df = load_results()
    print(f'  ✓ {len(df)} result rows', file=sys.stderr)

    if ensure_columns(df):
        atomic_pickle_dump(df, ANNOTATED_FILE)
        print('  ✓ Added missing columns', file=sys.stderr)
    sys.stderr.flush()

    done_ids = set(_normalize_uniprot_ids(load_progress()))
    all_ids = set(_normalize_uniprot_ids(
        df['target_uniprot_id'].tolist() + df['query_uniprot_id'].tolist()
    ))
    pending_ids = all_ids - done_ids
    print(f'\nTotal IDs: {len(all_ids):,} | done: {len(done_ids):,} | pending: {len(pending_ids):,}', file=sys.stderr)

    query_map  = df.groupby('query_uniprot_id').groups
    target_map = df.groupby('target_uniprot_id').groups
    dimer_map  = df.groupby('target_dimer_uniprot_id').groups
    print(f'Annotating {len(pending_ids):,} entries...\n', file=sys.stderr)
    sys.stderr.flush()

    pending_list = sorted(pending_ids)
    batch_num = 0
    total_pending = len(pending_list)
    attempted_total = 0
    found_total = 0
    pct_bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    overall_pbar = tqdm(
        total=total_pending,
        desc='Overall annotation',
        unit='id',
        file=sys.stderr,
        bar_format=pct_bar_format,
    )

    for cache, remaining in fetch_entries_batch(pending_list):
        check_memory_safety(f'batch {batch_num}')
        batch_num += 1

        batch_start = attempted_total
        batch_attempted = total_pending - len(remaining) - attempted_total
        if batch_attempted < 0:
            batch_attempted = 0
        attempted_total += batch_attempted
        batch_end = attempted_total
        attempted_ids = pending_list[batch_start:batch_end]

        mem_gb = ram_used_gb() or 0
        batch_ids = list(cache.keys())
        found_in_batch = len(batch_ids)
        missing_in_batch = max(0, batch_attempted - found_in_batch)
        found_total += found_in_batch
        print(
            f'Batch {batch_num}: found {found_in_batch:,}/{batch_attempted:,} IDs '
            f'(missing {missing_in_batch:,}) [{mem_gb:.1f}/{_TOTAL_MEM_GB:.0f}GB RAM]',
            file=sys.stderr,
        )
        sys.stderr.flush()

        pbar = tqdm(
            attempted_ids,
            total=batch_attempted,
            desc=f'Batch {batch_num}',
            leave=False,
            unit='id',
            file=sys.stderr,
            bar_format=pct_bar_format,
        )
        for i, uniprot_id in enumerate(pbar, 1):
            info = cache.get(uniprot_id)
            if info:
                annotate_id(df, uniprot_id, info, query_map, target_map, dimer_map)

            overall_pbar.update(1)
            if i % 1000 == 0:
                mem_gb = ram_used_gb() or 0
                pbar.set_postfix_str(f'{mem_gb:.1f}/{BATCH_RAM_LIMIT_GB:.1f}GB RAM limit')
                overall_pbar.set_postfix_str(
                    f'found {found_total:,} | missing {max(0, (batch_start + i) - found_total):,}'
                )

        overall_pbar.set_postfix_str(
            f'found {found_total:,} | missing {max(0, attempted_total - found_total):,}'
        )

        # Mark all attempted IDs as processed, including IDs not found in UniProt DB.
        if attempted_ids:
            done_ids.update(attempted_ids)

        # Checkpoint after each batch (batch size is already RAM-limited)
        atomic_pickle_dump(df, ANNOTATED_FILE)
        atomic_pickle_dump(pd.DataFrame({'uniprot_id': sorted(done_ids)}), PROGRESS_FILE)

        del cache
        gc.collect()

    overall_pbar.close()

    # Final save
    print('\nSaving final results...', file=sys.stderr)
    atomic_pickle_dump(df, ANNOTATED_FILE)
    atomic_pickle_dump(pd.DataFrame({'uniprot_id': sorted(done_ids)}), PROGRESS_FILE)
    print(f'  ✓ {ANNOTATED_FILE}', file=sys.stderr)

    # Cleanup
    for _, xml_file, _ in UNIPROT_SOURCES:
        if os.path.exists(xml_file):
            os.remove(xml_file)

    print('✓ Complete!', file=sys.stderr)


if __name__ == '__main__':
    main()
