# stdlib
import gc
import gzip
import io
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
try:
    from lxml import etree
    _XMLSyntaxError = etree.XMLSyntaxError
    _USING_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    _XMLSyntaxError = etree.ParseError
    _USING_LXML = False
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,  # kept for potential use
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,

)

class MofNCommaColumn(ProgressColumn):
    """M of N progress column with comma-separated thousands."""

    def render(self, task: Task) -> 'Text':
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else None
        if total is not None:
            return Text(f'{completed:,}/{total:,}', style='progress.download')
        return Text(f'{completed:,}', style='progress.download')


class EntriesPerSecondColumn(ProgressColumn):
    """Shows processing speed in entries/s."""

    def render(self, task: Task) -> 'Text':
        speed = task.speed
        if speed is None:
            return Text('— entries/s', style='progress.data.speed')
        if speed >= 1000:
            return Text(f'{speed/1000:.1f}k entries/s', style='progress.data.speed')
        return Text(f'{speed:.1f} entries/s', style='progress.data.speed')


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
    """Get current process RSS memory usage in GB."""
    if HAS_PSUTIL and psutil is not None:
        return psutil.Process().memory_info().rss / (1024 ** 3)

    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    return rss_kb / (1024 ** 2)
    except (OSError, KeyError, ValueError):
        pass
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
                # uniprot_dict values are pre-normalized (already lowercase)
                if anc not in seen:
                    seen.add(anc)
                    expanded.append(anc)
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
        # Expand through the UniProt subcell hierarchy (includes self + ancestors).
        # Values in uniprot_dict are pre-normalized, so no need to normalize again.
        candidates = uniprot_dict.get(normalized, [normalized])
        for t in candidates:
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


_local_cache: dict[str, str] = {}

def _local(tag):
    """Return the local part of a namespace-qualified XML tag."""
    try:
        return _local_cache[tag]
    except KeyError:
        result = tag.split('}')[-1] if '}' in tag else tag
        _local_cache[tag] = result
        return result


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
    protein_name_parents = {'recommendedName', 'submittedName', 'alternativeName'}

    # Iterate direct children of root to avoid building a parent map.
    # UniProt XML structure is well-defined: top-level children are protein,
    # gene, organism, dbReference, comment, feature, etc.
    for child in root:
        child_local = _local(child.tag)

        # Protein name — walk recommendedName / submittedName / alternativeName
        if child_local == 'protein' and not result['protein_name']:
            for name_el in child:
                if _local(name_el.tag) in protein_name_parents:
                    fn = next((c for c in name_el if _local(c.tag) == 'fullName'), None)
                    if fn is not None and fn.text:
                        result['protein_name'] = fn.text.strip()
                        break

        # Gene name
        elif child_local == 'gene' and not result['gene_name']:
            for name_el in child:
                if _local(name_el.tag) == 'name' and name_el.text:
                    result['gene_name'] = name_el.text.strip()
                    break

        # Organism
        elif child_local == 'organism' and not result['organism']:
            for name_el in child:
                if _local(name_el.tag) == 'name' and name_el.get('type') == 'scientific' and name_el.text:
                    result['organism'] = name_el.text.strip().lower()
                    break

        # Database references (Ensembl, PANTHER, GO)
        elif child_local == 'dbReference':
            dbtype = child.get('type')
            dbid = child.get('id')
            if dbtype and dbid:
                if dbtype == 'Ensembl' and not result['ensembl_id']:
                    result['ensembl_id'] = dbid.split('.')[0]
                elif dbtype == 'PANTHER':
                    result['panthr_id'].append(dbid)
                elif dbtype == 'GO':
                    for prop in child:
                        if _local(prop.tag) == 'property':
                            val = prop.get('value') or (prop.text.strip() if prop.text else '')
                            if val.startswith('C:'):
                                term = val[2:]
                                if term not in seen_go_terms:
                                    seen_go_terms.add(term)
                                    result['localization_go'].append({'go_id': dbid, 'term': term})

        # Subcellular location comments
        elif child_local == 'comment' and child.get('type') == 'subcellular location':
            for subcell in child:
                if _local(subcell.tag) not in ('subcellularLocation', 'subcellularLocations'):
                    continue
                for loc in subcell:
                    if _local(loc.tag) == 'location':
                        v = next((c for c in loc if _local(c.tag) == 'value'), None)
                        val = (v.text.strip() if v is not None and v.text else None) or \
                              (loc.text.strip() if loc.text else None)
                        if val and val not in result['localization_uniprot']:
                            result['localization_uniprot'].append(val)

        # Features (domains, etc.)
        elif child_local == 'feature':
            ftype = child.get('type', '').lower()
            if ftype not in domain_types:
                continue

            loc = next((c for c in child if _local(c.tag) == 'location'), None)
            if loc is not None:
                begin = _extract_pos(next((c for c in loc if _local(c.tag) == 'begin'), None))
                end = _extract_pos(next((c for c in loc if _local(c.tag) == 'end'), None))
                position = _extract_pos(next((c for c in loc if _local(c.tag) == 'position'), None))

                start = begin if begin is not None else position
                end_val = end if end is not None else position

                if start is not None and end_val is not None:
                    result['domains'].append({
                        'type': child.get('type', ''),
                        'description': child.get('description', ''),
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



def _source_tmp_path(db_file, source_name):
    """Return the per-source tmp DB path, e.g. uniprot_index_swissprot.tmp.db"""
    base, ext = os.path.splitext(db_file)
    slug = source_name.lower().replace('-', '')  # 'swissprot' or 'trembl'
    return f'{base}_{slug}.tmp{ext}'


def _open_xml_source(url, local_path, source_name, has_pigz):
    """Return a binary stream of decompressed UniProt XML.

    For local files: uses pigz (parallel, fast) when available.
    For streaming: uses curl → Python gzip.open (pigz on a pipe is unreliable
    because lxml closing the read end mid-stream causes SIGPIPE in pigz/curl).
    lxml.iterparse requires a binary stream; gzip.open('rb') returns bytes.
    """
    _BUFSIZE = 1 << 23  # 8 MB read buffer

    if os.path.exists(local_path):
        size_gb = os.path.getsize(local_path) / (1024 ** 3)
        if has_pigz:
            _console.print(f'  Using local file: [green]{local_path}[/] ({size_gb:.1f} GB) via [bold]pigz[/]')
            proc = subprocess.Popen(
                ['pigz', '-d', '-c', local_path],
                stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=_BUFSIZE,
            )
            return proc.stdout
        _console.print(f'  Using local file: [green]{local_path}[/] ({size_gb:.1f} GB)')
        return gzip.open(local_path, 'rb')

    # Streaming: curl → gzip.open (Python's gzip handles multi-member streams
    # correctly and doesn't suffer from SIGPIPE issues with lxml).
    _console.print(f'  Streaming [bold]{source_name}[/] directly from UniProt FTP…')
    proc = subprocess.Popen(
        ['curl', '-L', '--retry', '10', '--retry-delay', '5',
         '--retry-max-time', '3600', '--silent', '--show-error', url],
        stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=_BUFSIZE,
    )
    gz = gzip.open(proc.stdout, 'rb')
    # Wrap in an anonymous RawIOBase so lxml cannot resolve a .name attribute
    # to a file path (an empty .name resolves to CWD, causing XMLSyntaxError).
    class _AnonRaw(io.RawIOBase):
        def readinto(self, b):
            # RawIOBase.readinto must return exactly 0 only on EOF,
            # never a short read mid-stream, or BufferedReader signals EOF.
            data = gz.read(len(b))
            if not data:
                return 0
            n = len(data)
            b[:n] = data
            return n
        def readable(self):
            return True
    return io.BufferedReader(_AnonRaw(), buffer_size=_BUFSIZE)


def setup_uniprot_database(go_dict, uniprot_dict, db_file=UNIPROT_DB_FILE):
    """Stream and index Swiss-Prot + TrEMBL into SQLite.

    Each source is written to its own per-source .tmp.db file and only promoted
    (kept) once that source completes. If a source was interrupted its .tmp.db is
    discarded and re-indexed. Once both sources are done they are merged into the
    final db_file and the per-source tmps are removed.
    """
    # Already fully built — nothing to do.
    if os.path.exists(db_file):
        _console.print('[bold green]✓[/] All sources already indexed — skipping setup_uniprot_database.')
        return

    _console.print(f'[bold cyan]Building UniProt database:[/] [green]{db_file}[/]')

    _HAS_PIGZ = subprocess.run(['which', 'pigz'], capture_output=True).returncode == 0
    if _HAS_PIGZ:
        _console.print('  [dim]pigz detected — using parallel decompression for local files[/dim]')
    if _USING_LXML:
        _console.print('  [dim]lxml detected — using fast XML parser[/dim]')

    _console.print('[dim]Fetching online UniProt entry counts…[/]')
    online_totals = _fetch_online_uniprot_totals()

    # Track per-source completed tmp paths for the final merge step.
    completed_tmps = {}  # source_name -> tmp_path

    for url, xml_file, reviewed_flag in UNIPROT_SOURCES:
        source_name  = 'Swiss-Prot' if reviewed_flag == 1 else 'TrEMBL'
        source_color = 'yellow' if reviewed_flag == 1 else 'cyan'
        total_entries = online_totals.get(reviewed_flag)
        tmp_path = _source_tmp_path(db_file, source_name)
        done_marker = tmp_path + '.done'

        if os.path.exists(done_marker) and os.path.exists(tmp_path):
            _console.print(f'  [bold green]✓[/] [{source_color}]{source_name}[/] already indexed — skipping.')
            completed_tmps[source_name] = tmp_path
            continue

        # Keep any partial tmp from a previous interrupted run — we resume from it.
        # Only discard the done_marker (it can't exist here since we passed the check above).
        if os.path.exists(done_marker):
            os.remove(done_marker)

        upsert_sql = (
            'INSERT OR IGNORE INTO entries(accession, reviewed, organism, data) VALUES (?,?,?,?)'
        )

        _MAX_RETRIES = 20
        _RETRY_DELAY = 30  # seconds between retries

        for _attempt in range(1, _MAX_RETRIES + 1):
            conn = sqlite3.connect(tmp_path)
            conn.execute('PRAGMA page_size = 16384')
            conn.execute(f'PRAGMA journal_mode = {PRAGMA_JOURNAL_MODE}')
            conn.execute(f'PRAGMA synchronous = {PRAGMA_SYNCHRONOUS}')
            conn.execute(f'PRAGMA cache_size = {PRAGMA_CACHE_SIZE}')
            conn.execute('PRAGMA temp_store = MEMORY')
            conn.execute('CREATE TABLE IF NOT EXISTS entries '
                         '(accession TEXT PRIMARY KEY, reviewed INTEGER NOT NULL, '
                         'organism TEXT NOT NULL, data BLOB NOT NULL)')
            conn.commit()

            # Count already-persisted rows so progress display is accurate on resume.
            already_done = conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]

            n, batch = already_done, []

            if already_done > 0:
                _console.print(
                    f'  [dim][{source_color}]{source_name}[/{source_color}] resuming from '
                    f'{already_done:,} already-indexed entries (attempt {_attempt}/{_MAX_RETRIES})…[/dim]'
                )

            src_file = _open_xml_source(url, xml_file, source_name, _HAS_PIGZ)

            index_progress = Progress(
                SpinnerColumn(),
                TextColumn(f'  [{source_color}][bold]{source_name}[/bold][/{source_color}]'),
                BarColumn(bar_width=30),
                MofNCommaColumn(),
                TaskProgressColumn(),
                EntriesPerSecondColumn(),
                TimeElapsedColumn(),
                TextColumn('[dim]eta[/dim]'),
                TimeRemainingColumn(),
                TextColumn('[dim]{task.fields[ram]}[/dim]'),
                console=_console,
                transient=False,
            )
            task_id = index_progress.add_task(source_name, total=total_entries, ram='decompressing…',
                                              completed=already_done)

            _stream_error = None
            try:
                with index_progress:
                    with src_file as f:
                        _iterparse = etree.iterparse(
                            f,
                            events=('end',),
                        )
                        for event, elem in _iterparse:
                            tag_local = _local(elem.tag)

                            if tag_local != 'entry':
                                continue

                            try:
                                # Accessions are always direct children of <entry> — no need to walk descendants
                                accs = [c.text for c in elem if _local(c.tag) == 'accession' and c.text]

                                if accs:
                                    parsed = parse_entry_xml(elem, go_dict, uniprot_dict)
                                    data_blob = pickle.dumps(parsed, protocol=5)
                                    organism = parsed.get('organism', '')
                                    # Only insert primary accession (accs[0]); secondary accessions
                                    # point to the same record — avoids duplicating the blob N times.
                                    batch.append((accs[0], reviewed_flag, organism, data_blob))
                                    n += 1
                                    if n == already_done + 1:
                                        index_progress.update(task_id, ram='RAM: —')
                                    index_progress.advance(task_id)

                                    if n % 10000 == 0:
                                        mem_gb = ram_used_gb()
                                        ram_str = (f'RAM: {mem_gb:.1f}/{_TOTAL_MEM_GB:.0f} GB'
                                                   if mem_gb else 'RAM: —')
                                        index_progress.update(task_id, ram=ram_str)
                                        check_memory_safety('indexing')
                                        if (mem_gb is not None and mem_gb >= FLUSH_RAM_LIMIT_GB) or len(batch) >= 50000:
                                            conn.executemany(upsert_sql, batch)
                                            conn.commit()
                                            batch = []
                                            gc.collect()
                            except Exception:
                                pass
                            finally:
                                # Clear processed entry element to free memory.
                                # Do NOT clear _doc_root[:] — removing children from the
                                # root while lxml is still parsing causes XMLSyntaxError.
                                elem.clear()
            except _XMLSyntaxError as exc:
                # lxml raises XMLSyntaxError when the underlying stream is truncated
                # (e.g. a broken pipe from curl/pigz). Flush whatever we have, then retry.
                _stream_error = exc
            finally:
                if batch:
                    conn.executemany(upsert_sql, batch)
                    conn.commit()
                    batch = []
                conn.close()
                gc.collect()

            if _stream_error is not None:
                _console.print(
                    f'  [bold yellow]⚠[/] [{source_color}]{source_name}[/{source_color}] stream '
                    f'truncated after {n:,} entries (lxml: {_stream_error}). '
                    f'Saved progress. Retrying in {_RETRY_DELAY}s '
                    f'(attempt {_attempt}/{_MAX_RETRIES})…'
                )
                import time as _time
                _time.sleep(_RETRY_DELAY)
                continue  # retry

            # No stream error — source completed successfully.
            break
        else:
            raise RuntimeError(
                f'{source_name} stream kept truncating after {_MAX_RETRIES} attempts. '
                f'Last known progress: {n:,} entries. Check network/disk and re-run.'
            )

        # Re-open to get final count for display
        _final_conn = sqlite3.connect(tmp_path)
        _final_n = _final_conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
        _final_conn.close()

        # Mark this source as complete with a sentinel file.
        open(done_marker, 'w').close()
        completed_tmps[source_name] = tmp_path
        _console.print(f'  [bold green]✓[/] [{source_color}]{source_name}[/]: [bold]{_final_n:,}[/] entries indexed.')
        safe_collect()

    # ── Verify all sources completed before merging ────────────────────────────
    expected_sources = {('Swiss-Prot' if reviewed == 1 else 'TrEMBL') for _, _, reviewed in UNIPROT_SOURCES}
    missing_sources = expected_sources - set(completed_tmps.keys())
    if missing_sources:
        raise RuntimeError(
            f'Cannot build final database: the following sources did not complete: '
            f'{", ".join(sorted(missing_sources))}. Re-run to retry.'
        )

    # ── Merge per-source tmp DBs into the final DB ─────────────────────────────
    _console.print('[dim]Merging sources into final database…[/dim]')
    merge_tmp = db_file + '.merge.tmp'
    if os.path.exists(merge_tmp):
        os.remove(merge_tmp)

    final_conn = sqlite3.connect(merge_tmp)
    final_conn.execute('PRAGMA page_size = 16384')
    final_conn.execute(f'PRAGMA journal_mode = {PRAGMA_JOURNAL_MODE}')
    final_conn.execute(f'PRAGMA synchronous = {PRAGMA_SYNCHRONOUS}')
    final_conn.execute(f'PRAGMA cache_size = {PRAGMA_CACHE_SIZE}')
    final_conn.execute('PRAGMA temp_store = MEMORY')
    final_conn.execute('CREATE TABLE entries '
                       '(accession TEXT PRIMARY KEY, reviewed INTEGER NOT NULL, '
                       'organism TEXT NOT NULL, data BLOB NOT NULL)')
    final_conn.commit()

    for source_name, tmp_path in completed_tmps.items():
        source_color = 'yellow' if source_name == 'Swiss-Prot' else 'cyan'
        _console.print(f'  Merging [{source_color}]{source_name}[/]…')
        final_conn.execute(f"ATTACH DATABASE '{tmp_path}' AS src")
        final_conn.execute(
            'INSERT OR IGNORE INTO entries SELECT accession, reviewed, organism, data FROM src.entries'
        )
        final_conn.commit()
        final_conn.execute('DETACH DATABASE src')

    final_conn.close()
    os.replace(merge_tmp, db_file)

    # Clean up per-source tmps now that the final DB is ready.
    for source_name, tmp_path in completed_tmps.items():
        done_marker = tmp_path + '.done'
        for f in (tmp_path, done_marker):
            if os.path.exists(f):
                os.remove(f)

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
        f'[bold]{_TOTAL_MEM_GB:.1f} GB[/] total RAM  '
        f'process flush @ [yellow]{FLUSH_RAM_PCT*100:.0f}%[/] ({FLUSH_RAM_LIMIT_GB:.1f} GB)  '
        f'process hard limit @ [red]{MEMORY_HARD_LIMIT_PCT*100:.0f}%[/] ({MEMORY_HARD_LIMIT_GB:.1f} GB)'
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
        BarColumn(bar_width=30),
        MofNCommaColumn(),
        TaskProgressColumn(),
        EntriesPerSecondColumn(),
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
        BarColumn(bar_width=25),
        MofNCommaColumn(),
        TaskProgressColumn(),
        EntriesPerSecondColumn(),
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
