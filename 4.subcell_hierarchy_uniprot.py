import os
import pickle
import re
import urllib.request
from collections import defaultdict
from urllib.error import HTTPError, URLError


SUBCELL_FILE = "subcell.txt"
GO_OBO_FILE = "go-basic.obo"
GO_HIERARCHY_FILE = "go_hierarchy.pkl"
SUBCELL_URL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/docs/subcell.txt"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"


def normalize_term(value):
    return value.strip().rstrip(".").lower()


def parse_subcell(file_path):
    graph = defaultdict(set)
    all_terms = set()
    current_id = None

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("ID"):
                current_id = normalize_term(raw[5:])
                if not current_id:
                    continue
                all_terms.add(current_id)
                graph.setdefault(current_id, set())
            elif (raw.startswith("HP") or raw.startswith("HI")) and current_id:
                content = raw[5:].strip().rstrip(".")
                if not content:
                    continue

                parents = []
                for parent_raw in content.split(";"):
                    parent = normalize_term(parent_raw)
                    if parent:
                        parents.append(parent)

                for parent in parents:
                    all_terms.add(parent)
                    graph[current_id].add(parent)
            elif raw.startswith("//"):
                current_id = None

    for term in all_terms:
        graph.setdefault(term, set())

    return graph


def parse_go_mapping(file_path):
    """Build GO ID -> normalized UniProt subcellular term mapping from subcell.txt."""
    go_dict = {}
    current_id = None
    go_pattern = re.compile(r"(GO:\d+)")

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("ID"):
                current_id = normalize_term(raw[5:])
            elif raw.startswith("GO") and current_id:
                match = go_pattern.search(raw)
                if match:
                    go_dict[match.group(1)] = current_id
            elif raw.startswith("//"):
                current_id = None

    return go_dict


def parse_go_obo(file_path):
    """Parse go-basic.obo into a GO parent graph, name map, and alt_id alias map."""
    graph = defaultdict(set)
    names = {}
    alt_id_to_primary = {}

    current = None
    stanza_type = None

    def flush_current():
        nonlocal current, stanza_type
        if stanza_type != "Term" or not current:
            return

        go_id = current.get("id")
        if not go_id:
            return

        if current.get("is_obsolete") == "true":
            return

        graph.setdefault(go_id, set())
        if current.get("name"):
            names[go_id] = current["name"]

        for parent in current.get("is_a", []):
            graph[go_id].add(parent)

        for rel_type, parent in current.get("relationship", []):
            if rel_type == "part_of":
                graph[go_id].add(parent)

        for alt_id in current.get("alt_id", []):
            alt_id_to_primary[alt_id] = go_id

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()

            if line == "[Term]":
                flush_current()
                current = {}
                stanza_type = "Term"
                continue

            if line.startswith("["):
                flush_current()
                current = None
                stanza_type = None
                continue

            if not line or line.startswith("!") or current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("alt_id: "):
                current.setdefault("alt_id", []).append(line[8:].strip())
            elif line.startswith("is_a: "):
                parent = line[6:].split(" !", 1)[0].strip()
                if parent:
                    current.setdefault("is_a", []).append(parent)
            elif line.startswith("relationship: "):
                rel = line[len("relationship: "):]
                match = re.match(r"(\S+)\s+(GO:\d+)", rel)
                if match:
                    current.setdefault("relationship", []).append((match.group(1), match.group(2)))
            elif line.startswith("is_obsolete: "):
                current["is_obsolete"] = line.split(":", 1)[1].strip().lower()

    flush_current()
    return graph, names, alt_id_to_primary


def get_all_ancestors(graph, term, visited=None):
    if visited is None:
        visited = set()
    if term in visited:
        return set()

    visited.add(term)
    all_ancestors = set()
    for parent in graph.get(term, set()):
        all_ancestors.add(parent)
        all_ancestors.update(get_all_ancestors(graph, parent, visited.copy()))
    return all_ancestors


def build_ancestors_dict(graph):
    ancestors_dict = {}
    for term in graph:
        normalized_term = normalize_term(term)
        if not normalized_term:
            continue

        direct_parents = set(graph.get(normalized_term, set()))
        ancestors = {normalized_term}
        ancestors.update(direct_parents)
        for parent in direct_parents:
            ancestors.update(get_all_ancestors(graph, parent))

        # Defensive guarantee: each term should always include itself.
        ancestors.add(normalized_term)
        ancestors_dict[normalized_term] = sorted(ancestors)

    return ancestors_dict


def build_go_hierarchy(file_path):
    graph, names, alt_id_to_primary = parse_go_obo(file_path)

    hierarchy = {}
    for go_id in graph:
        ancestors = {go_id}
        direct_parents = set(graph.get(go_id, set()))
        ancestors.update(direct_parents)
        for parent in direct_parents:
            ancestors.update(get_all_ancestors(graph, parent))
        hierarchy[go_id] = [
            {
                "go_id": ancestor_id,
                "term": names.get(ancestor_id, ancestor_id),
            }
            for ancestor_id in sorted(ancestors)
        ]

    for alt_id, primary_id in alt_id_to_primary.items():
        if primary_id in hierarchy:
            hierarchy[alt_id] = [dict(item) for item in hierarchy[primary_id]]

    return hierarchy


def download_with_fallbacks(urls, file_path):
    last_error = None
    for url in urls:
        temp_path = f"{file_path}.part"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; foldseek_2/1.0)"},
            )
            with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out:
                out.write(response.read())

            os.replace(temp_path, file_path)
            return url
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            if os.path.exists(temp_path):
                os.remove(temp_path)

    raise RuntimeError(
        f"Failed to download {file_path} from all sources: {', '.join(urls)}"
    ) from last_error


def ensure_required_files():
    required = [
        (SUBCELL_FILE, [SUBCELL_URL], False),
        (GO_OBO_FILE, [GO_OBO_URL], False),
    ]
    for file_path, urls, optional in required:
        if os.path.exists(file_path):
            continue

        print(f"Missing {file_path}; downloading from {urls[0]}")
        try:
            downloaded_from = download_with_fallbacks(urls, file_path)
            print(f"Downloaded {file_path} from {downloaded_from}")
        except RuntimeError as exc:
            if optional:
                print(f"Warning: {exc}. Continuing without {file_path}.")
            else:
                raise


def main():
    ensure_required_files()

    graph = parse_subcell(SUBCELL_FILE)
    ancestors_dict = build_ancestors_dict(graph)
    with open("subcell_hierarchy.pkl", "wb") as f:
        pickle.dump(ancestors_dict, f)

    go_hierarchy = build_go_hierarchy(GO_OBO_FILE)
    with open(GO_HIERARCHY_FILE, "wb") as f:
        pickle.dump(go_hierarchy, f)

    go_dict = parse_go_mapping(SUBCELL_FILE)
    with open("go_dict.pkl", "wb") as f:
        pickle.dump(go_dict, f)

    print(f"Saved subcell_hierarchy.pkl ({len(ancestors_dict):,} terms)")
    print(f"Saved {GO_HIERARCHY_FILE} ({len(go_hierarchy):,} GO terms)")
    print(f"Saved go_dict.pkl ({len(go_dict):,} GO mappings)")


if __name__ == "__main__":
    main()
