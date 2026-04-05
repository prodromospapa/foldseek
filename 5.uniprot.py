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
import time
import urllib.parse
import urllib.request
from collections import deque

# third-party
import pandas as pd
import wget
import xml.etree.ElementTree as etree
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

_console = Console(stderr=True)

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
# Memory thresholds (as fraction of total RAM)
FLUSH_RAM_PCT         = 0.20   # Flush to DB when RAM usage hits this
MEMORY_HARD_LIMIT_PCT = 0.40   # Crash if usage exceeds this

# SQLite configuration
SQLITE_PARAM_LIMIT  = 900
PRAGMA_CACHE_SIZE   = -1000000
PRAGMA_JOURNAL_MODE = 'WAL'
PRAGMA_SYNCHRONOUS  = 'OFF'

UNIPROT_SOURCES = [
    ('https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz',
     'uniprot_sprot.xml.gz', 1),  # reviewed (Swiss-Prot)
    ('https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.xml.gz',
     'uniprot_trembl.xml.gz', 0),  # unreviewed (TrEMBL)
]


def _get_total_memory_gb():
    if HAS_PSUTIL and psutil is not None:
        return psutil.virtual_memory().total / (1024 ** 3)
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


def ensure_subcell_assets():
    """Ensure required subcell ontology assets exist; build them via 5.1 if needed."""
    required = ['subcell.txt', 'subcell_hierarchy.pkl', 'go_dict.pkl']
    missing = [p for p in required if not os.path.exists(p)]
    if not missing:
        return

    _console.print(f"[yellow]Missing ontology assets:[/] {', '.join(missing)}")
    _console.print('Running [bold]5.subcell_hierarchy_uniprot.py[/bold] to build missing assets…')
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


def _translate_go_to_uniprot(go_entries, go_dict):
    """Translate GO IDs to UniProt subcellular terms using go_dict. Untranslatable terms are discarded."""
    translated = []
    seen = set()
    for go in go_entries:
        go_id = go.get('go_id')
        if not go_id:
            continue
        uniprot_term = go_dict.get(go_id)
        if not uniprot_term:
            continue
        normalized = normalize_term(uniprot_term)
        if normalized and normalized not in seen:
            seen.add(normalized)
            translated.append(normalized)
    return translated


def _merge_localization_ancestors(uniprot_terms, translated_go_terms, uniprot_dict):
    """Merge UniProt and GO-translated-to-UniProt terms, expand through subcell hierarchy, deduplicate."""
    seen = set()
    merged = []

    all_terms = list(uniprot_terms or []) + list(translated_go_terms or [])
    for term in all_terms:
        normalized = normalize_term(term)
        if not normalized:
            continue
        # Expand through the UniProt subcell hierarchy (includes self + ancestors)
        candidates = uniprot_dict.get(normalized, [normalized])
        for t in candidates:
            t = normalize_term(t)
            if t and t not in seen:
                seen.add(t)
                merged.append(t)

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


def parse_entry_xml(root, go_dict, uniprot_dict):
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
                if parent_local in ('recommendedName', 'submittedName', 'alternativeName'):
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
    translated_go = _translate_go_to_uniprot(result['localization_go'], go_dict)
    result['localization_ancestors'] = _merge_localization_ancestors(
        result['localization_uniprot'],
        translated_go,
        uniprot_dict,
    )

    return result


def _is_source_indexed(conn, source_name):
    """Return True if source_name has been fully indexed in this DB."""
    conn.execute(
        'CREATE TABLE IF NOT EXISTS source_progress '
        '(source_name TEXT PRIMARY KEY, completed_at TEXT NOT NULL)'
    )
    conn.commit()
    row = conn.execute(
        'SELECT completed_at FROM source_progress WHERE source_name = ?', (source_name,)
    ).fetchone()
    return row is not None


def _mark_source_indexed(conn, source_name):
    """Record that source_name has been fully indexed."""
    import datetime
    conn.execute(
        'INSERT OR REPLACE INTO source_progress(source_name, completed_at) VALUES (?, ?)',
        (source_name, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()


def setup_uniprot_database(go_dict, uniprot_dict, db_file=UNIPROT_DB_FILE):
    """Stream and index Swiss-Prot + TrEMBL into SQLite."""
    conn = sqlite3.connect(db_file)
    conn.execute(f'PRAGMA journal_mode = {PRAGMA_JOURNAL_MODE}')
    conn.execute(f'PRAGMA synchronous = {PRAGMA_SYNCHRONOUS}')
    conn.execute(f'PRAGMA cache_size = {PRAGMA_CACHE_SIZE}')
    conn.execute('PRAGMA temp_store = MEMORY')
    conn.execute('CREATE TABLE IF NOT EXISTS entries '
                 '(accession TEXT PRIMARY KEY, reviewed INTEGER NOT NULL, organism TEXT NOT NULL, data BLOB NOT NULL)')
    conn.commit()

    _console.print(f'[bold cyan]Building UniProt database:[/] [green]{db_file}[/]')

    upsert_sql = (
        'INSERT OR REPLACE INTO entries(accession, reviewed, organism, data) VALUES (?,?,?,?)'
    )

    all_done = all(_is_source_indexed(conn, 'Swiss-Prot' if f == 1 else 'TrEMBL') for _, _, f in UNIPROT_SOURCES)
    if all_done:
        _console.print('[bold green]✓[/] All sources already indexed — skipping setup_uniprot_database.')
        conn.close()
        return

    _console.print('[dim]Fetching online UniProt entry counts…[/]')
    online_totals = _fetch_online_uniprot_totals()

    for url, xml_file, reviewed_flag in UNIPROT_SOURCES:
        source_name  = 'Swiss-Prot' if reviewed_flag == 1 else 'TrEMBL'
        source_color = 'yellow' if reviewed_flag == 1 else 'cyan'
        total_entries = online_totals.get(reviewed_flag)

        if _is_source_indexed(conn, source_name):
            _console.print(f'  [bold green]✓[/] [{source_color}]{source_name}[/] already indexed — skipping.')
            continue

        def _open_source(url=url, local_path=xml_file):
            if os.path.exists(local_path):
                size_gb = os.path.getsize(local_path) / (1024 ** 3)
                _console.print(f'  Using local file: [green]{local_path}[/] ({size_gb:.1f} GB)')
                return gzip.open(local_path, 'rt', encoding='utf-8', errors='ignore')

            # Stream via curl stdout → gzip → iterparse (no disk, starts immediately).
            _console.print(f'  Streaming [bold]{source_name}[/] directly from UniProt FTP…')
            proc = subprocess.Popen(
                ['curl', '-L', '--retry', '10', '--retry-delay', '5',
                 '--retry-max-time', '3600', '--silent', '--show-error', url],
                stdout=subprocess.PIPE, stderr=sys.stderr,
            )
            return gzip.open(proc.stdout, 'rt', encoding='utf-8', errors='ignore')

        n, batch = 0, []

        src_file = _open_source()

        index_progress = Progress(
            SpinnerColumn(),
            TextColumn(f'  [{source_color}][bold]{source_name}[/bold][/{source_color}]'),
            BarColumn(bar_width=45),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn('[dim]eta[/dim]'),
            TimeRemainingColumn(),
            TextColumn('[dim]{task.fields[ram]}[/dim]'),
            console=_console,
            transient=False,
        )
        task_id = index_progress.add_task(source_name, total=total_entries, ram='decompressing…')

        try:
            with index_progress:
                with src_file as f:
                    for event, elem in etree.iterparse(f, events=('end',)):
                        if not (elem.tag.endswith('}entry') or elem.tag == 'entry'):
                            continue

                        try:
                            accs = [acc.text for acc in elem.iter() if
                                   (acc.tag.endswith('}accession') or acc.tag == 'accession') and acc.text]

                            if accs:
                                parsed = parse_entry_xml(elem, go_dict, uniprot_dict)
                                data_blob = pickle.dumps(parsed, protocol=5)
                                organism = parsed.get('organism', '')
                                batch.extend((acc, reviewed_flag, organism, data_blob) for acc in accs)
                                n += 1
                                if n == 1:
                                    index_progress.update(task_id, ram='RAM: —')
                                index_progress.advance(task_id)

                                if n % 10000 == 0:
                                    mem_gb = ram_used_gb()
                                    ram_str = (f'RAM: {mem_gb:.1f}/{_TOTAL_MEM_GB:.0f} GB'
                                               if mem_gb else 'RAM: —')
                                    index_progress.update(task_id, ram=ram_str)
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
        finally:
            safe_collect()

        if batch:
            conn.executemany(upsert_sql, batch)
            conn.commit()
        _mark_source_indexed(conn, source_name)
        _console.print(f'  [bold green]✓[/] [{source_color}]{source_name}[/]: [bold]{n:,}[/] entries indexed.')
        safe_collect()

    conn.execute('CREATE INDEX IF NOT EXISTS idx_accession ON entries(accession)')
    conn.commit()

    conn.close()
    _console.print('[bold green]✓[/] UniProt database ready.')


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
                _console.print('[yellow]Detected legacy results file; rebuilding from base results.[/]')
                df = base_df.copy()
            del base_df
        except (EOFError, pickle.UnpicklingError):
            _console.print('[red]Corrupted annotated results file; rebuilding.[/]')
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
        _console.print(f'[dim]No progress checkpoint found ({PROGRESS_FILE}); starting fresh.[/dim]')
        return []
    try:
        return pd.read_pickle(PROGRESS_FILE)['uniprot_id'].tolist()
    except (EOFError, pickle.UnpicklingError):
        _console.print('[red]Corrupted progress file; restarting progress tracking.[/]')
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
    from rich.rule import Rule

    _console.print(Rule('[bold blue]UniProt Annotation Pipeline[/bold blue]'))

    # ── Ontology assets ────────────────────────────────────────────────────────
    _console.print('\n[bold]Step 1/3[/bold]  Loading ontology resources…')
    ensure_subcell_assets()

    uniprot_dict = pd.read_pickle('subcell_hierarchy.pkl')
    _console.print(f'  [green]✓[/] Subcellular terms: [bold]{len(uniprot_dict):,}[/]')

    go_dict = load_go_dict()
    _console.print(f'  [green]✓[/] GO→UniProt mappings: [bold]{len(go_dict):,}[/]')

    mem_info = (
        f'[bold]{_TOTAL_MEM_GB:.1f} GB[/] RAM  '
        f'flush @ [yellow]{FLUSH_RAM_PCT*100:.0f}%[/] ({FLUSH_RAM_LIMIT_GB:.1f} GB)  '
        f'hard limit @ [red]{MEMORY_HARD_LIMIT_PCT*100:.0f}%[/] ({MEMORY_HARD_LIMIT_GB:.1f} GB)'
    )
    _console.print(f'\n  [dim]System:[/dim] {mem_info}')

    # ── Database setup ─────────────────────────────────────────────────────────
    _console.print(f'\n[bold]Step 2/3[/bold]  Building UniProt SQLite database…')
    setup_uniprot_database(go_dict, uniprot_dict)

    # ── Annotation ─────────────────────────────────────────────────────────────
    _console.print(f'\n[bold]Step 3/3[/bold]  Annotating FoldSeek results…')
    df = load_results()
    _console.print(f'  [green]✓[/] Loaded [bold]{len(df):,}[/] result rows')

    if ensure_columns(df):
        atomic_pickle_dump(df, ANNOTATED_FILE)
        _console.print('  [green]✓[/] Added missing annotation columns')

    done_ids = set(_normalize_uniprot_ids(load_progress()))
    all_ids = set(_normalize_uniprot_ids(
        df['target_uniprot_id'].tolist()
        + df['query_uniprot_id'].tolist()
        + df['target_dimer_uniprot_id'].tolist()
    ))
    pending_ids = all_ids - done_ids

    _console.print(
        f'  Total IDs: [bold]{len(all_ids):,}[/]  '
        f'[green]done: {len(done_ids):,}[/]  '
        f'[yellow]pending: {len(pending_ids):,}[/]'
    )

    query_map  = df.groupby('query_uniprot_id').groups
    target_map = df.groupby('target_uniprot_id').groups
    dimer_map  = df.groupby('target_dimer_uniprot_id').groups

    pending_list  = sorted(pending_ids)
    batch_num     = 0
    total_pending = len(pending_list)
    attempted_total = 0
    found_total     = 0

    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn('[bold magenta]Overall[/bold magenta]'),
        BarColumn(bar_width=45),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn('[dim]eta[/dim]'),
        TimeRemainingColumn(),
        TextColumn('[dim]{task.fields[status]}[/dim]'),
        console=_console,
        transient=False,
    )
    batch_progress = Progress(
        SpinnerColumn(),
        TextColumn('  [cyan]{task.description}[/cyan]'),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn('[dim]{task.fields[ram]}[/dim]'),
        console=_console,
        transient=True,
    )

    overall_task = overall_progress.add_task(
        'annotation', total=total_pending, status='starting…'
    )

    _console.print()
    with overall_progress:
        for cache, remaining in fetch_entries_batch(pending_list):
            check_memory_safety(f'batch {batch_num}')
            batch_num += 1

            batch_start = attempted_total
            batch_attempted = total_pending - len(remaining) - attempted_total
            if batch_attempted < 0:
                batch_attempted = 0
            attempted_total += batch_attempted
            attempted_ids = pending_list[batch_start:attempted_total]

            mem_gb = ram_used_gb() or 0
            found_in_batch = len(cache)
            missing_in_batch = max(0, batch_attempted - found_in_batch)
            found_total += found_in_batch

            _console.print(
                f'  [bold]Batch {batch_num}[/]:  '
                f'[green]found {found_in_batch:,}[/] / {batch_attempted:,}  '
                f'[red]missing {missing_in_batch:,}[/]  '
                f'[dim]RAM {mem_gb:.1f}/{_TOTAL_MEM_GB:.0f} GB[/]'
            )

            batch_task = batch_progress.add_task(
                f'Batch {batch_num}', total=batch_attempted, ram='RAM: —'
            )

            with batch_progress:
                for i, uniprot_id in enumerate(attempted_ids, 1):
                    info = cache.get(uniprot_id)
                    if info:
                        annotate_id(df, uniprot_id, info, query_map, target_map, dimer_map)

                    batch_progress.advance(batch_task)
                    overall_progress.advance(overall_task)

                    if i % 1000 == 0:
                        mem_gb = ram_used_gb() or 0
                        batch_progress.update(
                            batch_task,
                            ram=f'RAM {mem_gb:.1f}/{BATCH_RAM_LIMIT_GB:.1f} GB',
                        )
                        overall_progress.update(
                            overall_task,
                            status=f'found {found_total:,} | missing {max(0, attempted_total - found_total):,}',
                        )

            overall_progress.update(
                overall_task,
                status=f'found {found_total:,} | missing {max(0, attempted_total - found_total):,}',
            )

            if attempted_ids:
                done_ids.update(attempted_ids)

            atomic_pickle_dump(df, ANNOTATED_FILE)
            atomic_pickle_dump(pd.DataFrame({'uniprot_id': sorted(done_ids)}), PROGRESS_FILE)

            del cache
            gc.collect()

    # ── Final save ─────────────────────────────────────────────────────────────
    _console.print('\n[dim]Saving final results…[/dim]')
    atomic_pickle_dump(df, ANNOTATED_FILE)
    atomic_pickle_dump(pd.DataFrame({'uniprot_id': sorted(done_ids)}), PROGRESS_FILE)
    _console.print(f'  [green]✓[/] [bold]{ANNOTATED_FILE}[/]')
    _console.print(Rule('[bold green]Complete![/bold green]'))


if __name__ == '__main__':
    main()
