#!/usr/bin/env python3
"""
Script to create a hierarchical tree of subcellular locations
based on UniProt's subcell.txt file and protein counts.
"""

import json
import re
import time
import urllib.parse
import urllib.request
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Parse subcell.txt to extract location hierarchy
def parse_subcell_file(filepath):
    """
    Parse subcell.txt and extract ID, HP (parent terms), and other info.
    Returns a dictionary with location info and their parent-child relationships.
    """
    locations = {}
    current_entry = {}
    first_entry_seen = False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        
        # Start of new entry
        if line.startswith('ID   '):
            # Save previous entry (skip only if it's the very first one)
            if current_entry and 'ID' in current_entry and first_entry_seen:
                term_id = current_entry['ID']
                locations[term_id] = current_entry
            
            term_name = line[5:].rstrip('.')
            current_entry = {'ID': term_name, 'parents': []}
            first_entry_seen = True
        
        elif line.startswith('HP   '):
            parent = line[5:].rstrip('.')
            current_entry.setdefault('parents', []).append(parent)
        
        elif line.startswith('//'):
            # End of entry - save if not the first one
            if current_entry and 'ID' in current_entry and first_entry_seen:
                term_id = current_entry['ID']
                # Don't save the very first entry (A band)
                if locations or term_id != 'A band':
                    locations[term_id] = current_entry
            current_entry = {}
    
    return locations

# Read organelle counts
def read_organelle_counts(filepath):
    """Read the organelle counts CSV file."""
    df = pd.read_csv(filepath)
    # Legacy aggregate format: Organelle, Count
    if 'Organelle' in df.columns and 'Count' in df.columns:
        counts = {}
        for _, row in df.iterrows():
            counts[str(row['Organelle']).lower()] = row['Count']
        return counts

    # New per-query matrix format: Query + compartment columns + Total
    ignore = {'Query', 'query', 'Total'}
    ignore.update({c for c in df.columns if str(c).startswith('Unnamed')})
    compartment_cols = [c for c in df.columns if c not in ignore]
    if not compartment_cols:
        raise ValueError(f'Unrecognized organelle counts format in {filepath}')

    counts = {}
    for col in compartment_cols:
        counts[str(col).lower()] = int(pd.to_numeric(df[col], errors='coerce').fillna(0).sum())
    return counts

# Count total proteins
def count_total_proteins(filepath):
    """Count the number of rows in the results file (excluding header)."""
    df = pd.read_csv(filepath)
    return len(df)

# UniProt totals per compartment
def fetch_uniprot_total_for_term(term_name, sleep_seconds=0.2):
    """Fetch total UniProt human proteins for a subcellular location term."""
    term = term_name.strip().replace('"', '\\"')
    query = f'(organism_id:9606) AND (cc_scl_term:"{term}")'
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "size": 0}
    )
    with urllib.request.urlopen(url) as response:
        response.read()
        total = response.headers.get("x-total-results")
    if total is None:
        raise ValueError(f"Unexpected UniProt response for term: {term_name}")
    time.sleep(sleep_seconds)
    return int(total)

def load_uniprot_totals(locations, cache_file='organelle_uniprot_totals.csv'):
    """Load UniProt totals from cache or fetch missing terms and update cache."""
    cache_path = Path(cache_file)
    totals = {}
    if cache_path.exists():
        df_cache = pd.read_csv(cache_path)
        for _, row in df_cache.iterrows():
            totals[row['subcellular_location'].lower()] = int(row['uniprot_human_proteins'])

    updated = False
    for _, info in locations.items():
        term_name = info['ID']
        key = term_name.lower()
        if key not in totals:
            totals[key] = fetch_uniprot_total_for_term(term_name)
            updated = True

    if updated or not cache_path.exists():
        df_out = pd.DataFrame(
            [
                {"subcellular_location": name, "uniprot_human_proteins": total}
                for name, total in sorted(totals.items())
            ]
        )
        df_out.to_csv(cache_path, index=False)

    return totals

# Build tree structure
def build_tree(locations, counts, total_proteins, uniprot_totals):
    """
    Build a tree structure with parent-child relationships.
    Returns a tree with nodes containing term info and counts.
    """
    tree = {
        'name': 'All proteins',
        'count': total_proteins,
        'percentage': 100.0,
        'uniprot_total': 0,
        'percent_of_uniprot': 0.0,
        'parents': [],
        'children': [],
        'parent_nodes': [],
        'child_nodes': []
    }
    
    # Dictionary to store nodes by name for easy lookup
    nodes = {'All proteins': tree}
    
    # Create nodes for all locations
    for term_id, info in locations.items():
        term_name = info['ID']
        term_key = term_name.lower()
        term_count = counts.get(term_key, 0)
        uniprot_total = uniprot_totals.get(term_key, 0)
        
        node = {
            'name': term_name,
            'term_id': term_id,
            'count': term_count,
            'percentage': (term_count / total_proteins * 100) if total_proteins > 0 else 0,
            'uniprot_total': uniprot_total,
            'percent_of_uniprot': (term_count / uniprot_total * 100) if uniprot_total > 0 else 0,
            'parents': info.get('parents', []),
            'children': [],
            'parent_nodes': [],  # Actual node references
            'child_nodes': []    # Actual node references
        }
        nodes[term_name] = node
    
    # Build parent-child relationships
    for term_name, node in nodes.items():
        if term_name == 'All proteins':
            continue
        
        if not node['parents']:
            # Root node - child of "All proteins"
            node['parent_nodes'].append(tree)
            tree['children'].append(node)
        else:
            # Has explicit parents
            for parent_name in node['parents']:
                if parent_name in nodes:
                    parent_node = nodes[parent_name]
                    node['parent_nodes'].append(parent_node)
                    if node not in parent_node['children']:
                        parent_node['children'].append(node)
    
    return tree, nodes

# Format tree output
def format_tree_output(node, indent=0, is_root=False, parent_is_primary=True):
    """
    Format the tree for display with hierarchy.
    """
    output = []
    prefix = "  " * indent
    
    if is_root:
        output.append(f"{prefix}{node['name']}")
        output.append(f"{prefix}Count: {node['count']}")
        output.append(f"{prefix}Percentage: {node['percentage']:.2f}%")
    else:
        parent_info = ""
        if len(node['parent_nodes']) > 1:
            if parent_is_primary:
                parent_info = f" [PRIMARY PARENT: {node['parent_nodes'][0]['name']}]"
                parent_info += f" [SECONDARY PARENT(S): {', '.join([p['name'] for p in node['parent_nodes'][1:]])}]"
            else:
                parent_info = f" [SECONDARY PARENT: {node['parent_nodes'][0]['name']}]"
        
        output.append(f"{prefix}├─ {node['name']}{parent_info}")
        output.append(
            f"{prefix}│  Count: {node['count']}, Percentage: {node['percentage']:.2f}%, "
            f"UniProt total: {node['uniprot_total']}, Homologues of UniProt: {node['percent_of_uniprot']:.2f}%"
        )
    
    # Sort children by count (descending) for better visualization
    # Filter out nodes with zero counts
    sorted_children = sorted([c for c in node['children'] if c['count'] > 0], 
                            key=lambda x: x['count'], reverse=True)
    
    for i, child in enumerate(sorted_children):
        is_last = i == len(sorted_children) - 1
        child_indent = indent + 1 if not is_root else indent + 1
        child_output = format_tree_output(child, child_indent, parent_is_primary=True)
        output.extend(child_output)
    
    return output

# Generate HTML tree visualization
def generate_html_tree(node, html_content=None, depth=0, node_id=None):
    """Generate HTML representation of the tree with expandable sections."""
    if html_content is None:
        html_content = []
    
    if node_id is None:
        node_id = [0]  # Use list to maintain counter across recursion
    
    current_id = node_id[0]
    node_id[0] += 1
    
    parent_info = ""
    if len(node['parent_nodes']) > 1 and depth > 0:
        parent_info = f"<br><small style='color:gray;'>" \
                      f"Primary Parent: {node['parent_nodes'][0]['name']}<br>" \
                      f"Secondary Parent(s): {', '.join([p['name'] for p in node['parent_nodes'][1:]])}" \
                      f"</small>"
    
    # Check if node has children with count > 0
    children_with_count = [c for c in node['children'] if c['count'] > 0]
    has_children = len(children_with_count) > 0
    
    if has_children:
        html_content.append(
            f"<div style='margin-left: {depth * 20}px; border-left: 2px solid #ccc; padding: 10px;'>"
            f"<button class='expand-btn' onclick='toggleExpand(this, {current_id})' style='background: none; border: none; cursor: pointer; color: #0066cc; font-weight: bold; padding: 0; margin-right: 5px;'>▶</button>"
            f"<span class='term-name' data-term='{node['name']}'><strong>{node['name']}</strong></span><br>"
            f"<span style='color: #666;'>Count: {node['count']} | "
            f"Percentage: {node['percentage']:.2f}% | "
            f"UniProt total: {node['uniprot_total']} | "
            f"Homologues of UniProt: {node['percent_of_uniprot']:.2f}%</span>"
            f"{parent_info}"
            f"<div id='node-{current_id}' style='display: none;'>"
        )
    else:
        html_content.append(
            f"<div style='margin-left: {depth * 20}px; border-left: 2px solid #ccc; padding: 10px;'>"
            f"<span class='term-name' data-term='{node['name']}'><strong>{node['name']}</strong></span><br>"
            f"<span style='color: #666;'>Count: {node['count']} | "
            f"Percentage: {node['percentage']:.2f}% | "
            f"UniProt total: {node['uniprot_total']} | "
            f"Homologues of UniProt: {node['percent_of_uniprot']:.2f}%</span>"
            f"{parent_info}"
        )
    
    # Sort children by count (descending) - Filter out nodes with zero counts
    sorted_children = sorted([c for c in node['children'] if c['count'] > 0], 
                            key=lambda x: x['count'], reverse=True)
    
    for child in sorted_children:
        generate_html_tree(child, html_content, depth + 1, node_id)
    
    if has_children:
        html_content.append("</div></div>")
    else:
        html_content.append("</div>")
    
    return html_content

def generate_statistics_dashboard(root_node, all_nodes, total_proteins, output_file='organelle_statistics_dashboard.html'):
    """Generate an interactive statistics dashboard with bar charts and data table."""
    
    # Collect all nodes with counts > 0
    nodes_with_data = []
    for node in all_nodes.values():
        if node['count'] > 0 and node['name'] != 'All proteins':
            nodes_with_data.append({
                'name': node['name'],
                'count': node['count'],
                'percentage': node['percentage'],
                'uniprot_total': node['uniprot_total'],
                'percent_of_uniprot': node['percent_of_uniprot'],
                'parents': ', '.join([p['name'] for p in node['parent_nodes']]) if node['parent_nodes'] else 'None',
                'children_count': len([c for c in node['children'] if c['count'] > 0])
            })
    
    # Calculate total proteins from actual node counts to ensure accuracy
    total_proteins_calculated = sum([node['count'] for node in nodes_with_data])
    
    # Sort by count descending
    nodes_with_data.sort(key=lambda x: x['count'], reverse=True)
    top_30 = nodes_with_data[:30]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Subcellular Location Statistics Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-section {{
            padding: 30px;
        }}
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .bar-chart {{
            margin: 30px 0;
        }}
        .bar-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 8px;
            border-radius: 4px;
            transition: background 0.2s;
        }}
        .bar-item:hover {{
            background: #f8f9fa;
        }}
        .bar-label {{
            width: 200px;
            font-size: 0.9em;
            padding-right: 15px;
            text-align: right;
            font-weight: 500;
            color: #333;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .bar-container {{
            flex: 1;
            height: 25px;
            background: #e9ecef;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.8s ease-out;
            display: flex;
            align-items: center;
            padding-right: 8px;
            justify-content: flex-end;
        }}
        .bar-value {{
            color: white;
            font-size: 0.8em;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            white-space: nowrap;
        }}
        .table-section {{
            padding: 30px;
            background: #f8f9fa;
        }}
        .table-controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .search-box {{
            flex: 1;
            min-width: 250px;
            padding: 12px 20px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .data-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        .data-table th:hover {{
            background: linear-gradient(135deg, #5568d3 0%, #653a8d 100%);
        }}
        .data-table th::after {{
            content: ' ⇅';
            opacity: 0.5;
            font-size: 0.8em;
        }}
        .data-table th.sorted-asc::after {{
            content: ' ↑';
            opacity: 1;
        }}
        .data-table th.sorted-desc::after {{
            content: ' ↓';
            opacity: 1;
        }}
        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .data-table tr:hover {{
            background: #f8f9fa;
        }}
        .data-table tr:last-child td {{
            border-bottom: none;
        }}
        .count-cell {{
            font-weight: bold;
            color: #667eea;
        }}
        .percentage-cell {{
            color: #28a745;
        }}
        .no-results {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Subcellular Location Statistics</h1>
            <p>Comprehensive analysis of {total_proteins_calculated:,} proteins across {len(nodes_with_data)} subcellular locations</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Proteins</div>
                <div class="stat-value">{total_proteins_calculated:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Locations</div>
                <div class="stat-value">{len(all_nodes) - 1}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">With Count Data</div>
                <div class="stat-value">{len(nodes_with_data)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Multi-Parent</div>
                <div class="stat-value">{len([n for n in all_nodes.values() if len(n.get('parent_nodes', [])) > 1])}</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="section-title">Top 30 Locations by Protein Count</h2>
            <div class="bar-chart">
"""
    
    # Add bar chart items
    max_count = top_30[0]['count'] if top_30 else 1
    for item in top_30:
        width_percent = (item['count'] / max_count) * 100
        html += f"""                <div class="bar-item">
                    <div class="bar-label" title="{item['name']}">{item['name']}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {width_percent}%">
                            <span class="bar-value">{item['count']:,} ({item['percentage']:.2f}%)</span>
                        </div>
                    </div>
                </div>
"""
    
    # Add data table
    html += """            </div>
        </div>
        
        <div class="table-section">
            <h2 class="section-title">All Locations - Interactive Table</h2>
            <div class="table-controls">
                <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search locations, parents..." />
            </div>
            <table class="data-table" id="dataTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Location</th>
                        <th onclick="sortTable(1)">Protein Count</th>
                        <th onclick="sortTable(2)">Percentage</th>
                        <th onclick="sortTable(3)">UniProt Total</th>
                        <th onclick="sortTable(4)">Homologues % of UniProt</th>
                        <th onclick="sortTable(5)">Parent(s)</th>
                        <th onclick="sortTable(6)">Children</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
"""
    
    for item in nodes_with_data:
        html += f"""                    <tr>
                        <td>{item['name']}</td>
                        <td class="count-cell">{item['count']:,}</td>
                        <td class="percentage-cell">{item['percentage']:.2f}%</td>
                        <td class="count-cell">{item['uniprot_total']:,}</td>
                        <td class="percentage-cell">{item['percent_of_uniprot']:.2f}%</td>
                        <td>{item['parents']}</td>
                        <td>{item['children_count']}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#tableBody tr');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        });
        
        // Table sorting
        let currentSort = {{ column: -1, ascending: true }};
        
        function sortTable(columnIndex) {{
            const table = document.getElementById('dataTable');
            const tbody = document.getElementById('tableBody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Toggle sort direction if same column
            if (currentSort.column === columnIndex) {{
                currentSort.ascending = !currentSort.ascending;
            }} else {{
                currentSort.column = columnIndex;
                currentSort.ascending = true;
            }}
            
            // Update header indicators
            table.querySelectorAll('th').forEach((th, idx) => {{
                th.classList.remove('sorted-asc', 'sorted-desc');
                if (idx === columnIndex) {{
                    th.classList.add(currentSort.ascending ? 'sorted-asc' : 'sorted-desc');
                }}
            }});
            
            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Handle numeric columns
                if (columnIndex === 1 || columnIndex === 3 || columnIndex === 6) {{
                    aVal = parseInt(aVal.replace(/,/g, ''));
                    bVal = parseInt(bVal.replace(/,/g, ''));
                }} else if (columnIndex === 2 || columnIndex === 4) {{
                    aVal = parseFloat(aVal);
                    bVal = parseFloat(bVal);
                }}
                
                if (aVal < bVal) return currentSort.ascending ? -1 : 1;
                if (aVal > bVal) return currentSort.ascending ? 1 : -1;
                return 0;
            }});
            
            // Reappend sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        // Animate bars on load
        window.addEventListener('load', function() {{
            const bars = document.querySelectorAll('.bar-fill');
            bars.forEach((bar, index) => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, index * 30);
            }});
        }});
    </script>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html)

# Main execution
def main():
    locations = parse_subcell_file('subcell.txt')
    
    # Process first file (original filtered)
    protein_counts_file = Path('per_query_organelle_counts_proteins.csv')
    if protein_counts_file.exists():
        counts = read_organelle_counts(str(protein_counts_file))
    else:
        counts = read_organelle_counts('organelle_counts_proteins.csv')
    
    total_proteins = count_total_proteins('results/foldseek_results_merged_with_localization_filtered.csv')
    total_proteins -= 1  # Subtract header row
    
    uniprot_totals = load_uniprot_totals(locations)
    root_node, all_nodes = build_tree(locations, counts, total_proteins, uniprot_totals)
    
    output_lines = format_tree_output(root_node, is_root=True)
    
    # Save to text file
    with open('organelle_hierarchy_tree.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUBCELLULAR LOCATION HIERARCHY\n")
        f.write("="*80 + "\n")
        for line in output_lines:
            f.write(line + "\n")
    
    # Generate and save HTML
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Subcellular Location Hierarchy</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
        "h1 { color: #333; }",
        ".search-container { margin-bottom: 20px; position: relative; width: 400px; }",
        ".search-input { width: 100%; padding: 10px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }",
        ".search-input:focus { outline: none; border-color: #0066cc; box-shadow: 0 0 5px rgba(0,102,204,0.3); }",
        ".suggestions { position: absolute; top: 100%; left: 0; width: 100%; background: white; border: 1px solid #ddd; border-top: none; max-height: 200px; overflow-y: auto; display: none; z-index: 1000; }",
        ".suggestion-item { padding: 10px; cursor: pointer; border-bottom: 1px solid #eee; }",
        ".suggestion-item:hover { background-color: #f0f0f0; }",
        ".suggestion-item.highlight { background-color: #e3f2fd; }",
        ".tree-container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
        ".legend { margin: 20px 0; font-size: 12px; }",
        ".legend-item { margin: 5px 0; }",
        ".expand-btn { color: #0066cc; font-size: 12px; }",
        ".term-name { user-select: none; }",
        ".term-name.highlighted { background-color: #ffeb3b; font-weight: bold; }",
        "small { font-size: 11px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Subcellular Location Hierarchy with Protein Localization</h1>",
        "<div class='search-container'>",
        "  <input type='text' id='searchInput' class='search-input' placeholder='Search organelles... (e.g., nucleus, mitochondrion)'>",
        "  <div id='suggestions' class='suggestions'></div>",
        "</div>",
        "<div class='legend'>",
        "<h3>Legend:</h3>",
        "<div class='legend-item'>• Count: Number of proteins localized to this term (from per_query_organelle_counts_proteins.csv, summed across queries)</div>",
        "<div class='legend-item'>• Percentage: Count as percentage of total proteins analyzed</div>",
        "<div class='legend-item'>• UniProt total: Total human proteins annotated to this term</div>",
        "<div class='legend-item'>• Homologues of UniProt: Count as percentage of UniProt total</div>",
        "<div class='legend-item'>• Primary Parent: Main parent location</div>",
        "<div class='legend-item'>• Secondary Parent(s): Additional parent locations (if term belongs to multiple categories)</div>",
        "<div class='legend-item'>• Click ▶ to expand/collapse child terms (closed by default for easier navigation)</div>",
        "</div>",
        "<div class='tree-container' id='treeContainer'>",
    ]
    
    html_lines.extend(generate_html_tree(root_node))
    
    html_lines.extend([
        "</div>",
        "<p style='font-size: 12px; color: #666; margin-top: 20px;'>",
        f"Total proteins analyzed: {total_proteins}<br>",
        f"Data sources: subcell.txt (UniProt), per_query_organelle_counts_proteins.csv, foldseek_results_merged_with_localization_filtered.csv",
        "</p>",
        "",
        "<script>",
        "// Collect all term names for autocomplete",
        "const allTerms = [];",
        "document.querySelectorAll('.term-name').forEach(el => {",
        "  allTerms.push(el.getAttribute('data-term'));",
        "});",
        "",
        "const searchInput = document.getElementById('searchInput');",
        "const suggestionsDiv = document.getElementById('suggestions');",
        "",
        "// Search and suggestion functionality",
        "searchInput.addEventListener('input', function() {",
        "  const query = this.value.toLowerCase().trim();",
        "  ",
        "  if (query.length === 0) {",
        "    suggestionsDiv.style.display = 'none';",
        "    return;",
        "  }",
        "  ",
        "  // Filter suggestions",
        "  const matches = allTerms.filter(term => ",
        "    term.toLowerCase().includes(query)",
        "  ).slice(0, 10);  // Limit to 10 suggestions",
        "  ",
        "  if (matches.length > 0) {",
        "    suggestionsDiv.innerHTML = matches.map((term, idx) => ",
        "      `<div class='suggestion-item' onclick='selectTerm(\"${term}\")' onmouseover='this.classList.add(\"highlight\")' onmouseout='this.classList.remove(\"highlight\")'>${term}</div>`",
        "    ).join('');",
        "    suggestionsDiv.style.display = 'block';",
        "  } else {",
        "    suggestionsDiv.style.display = 'none';",
        "  }",
        "});",
        "",
        "function selectTerm(termName) {",
        "  searchInput.value = termName;",
        "  suggestionsDiv.style.display = 'none';",
        "  highlightAndNavigate(termName);",
        "}",
        "",
        "function highlightAndNavigate(termName) {",
        "  // Remove previous highlighting",
        "  document.querySelectorAll('.term-name').forEach(el => {",
        "    el.classList.remove('highlighted');",
        "  });",
        "  ",
        "  // Find and highlight the matching term",
        "  const termElements = document.querySelectorAll('.term-name');",
        "  let found = false;",
        "  ",
        "  termElements.forEach(el => {",
        "    if (el.getAttribute('data-term').toLowerCase() === termName.toLowerCase()) {",
        "      el.classList.add('highlighted');",
        "      found = true;",
        "      ",
        "      // Expand all parent containers",
        "      let parent = el.closest('div[id^=\"node-\"]');",
        "      while (parent) {",
        "        parent.style.display = 'block';",
        "        const btn = parent.parentElement.querySelector('.expand-btn');",
        "        if (btn) {",
        "          btn.textContent = '▼';",
        "        }",
        "        parent = parent.parentElement.closest('div[id^=\"node-\"]');",
        "      }",
        "      ",
        "      // Scroll to element",
        "      el.scrollIntoView({ behavior: 'smooth', block: 'center' });",
        "    }",
        "  });",
        "  ",
        "  if (!found) {",
        "    alert('Term \"' + termName + '\" not found.');",
        "  }",
        "}",
        "",
        "function toggleExpand(btn, nodeId) {",
        "  const container = document.getElementById('node-' + nodeId);",
        "  if (container.style.display === 'none') {",
        "    container.style.display = 'block';",
        "    btn.textContent = '▼';",
        "  } else {",
        "    container.style.display = 'none';",
        "    btn.textContent = '▶';",
        "  }",
        "}",
        "",
        "// Close suggestions when clicking outside",
        "document.addEventListener('click', function(e) {",
        "  if (e.target !== searchInput) {",
        "    suggestionsDiv.style.display = 'none';",
        "  }",
        "});",
        "</script>",
        "</body>",
        "</html>"
    ])
    
    with open('organelle_hierarchy_tree.html', 'w') as f:
        f.write("\n".join(html_lines))
    
    # Generate statistics dashboard
    generate_statistics_dashboard(root_node, all_nodes, total_proteins)
    
    # Check if unique genes file exists and process it
    import os
    unique_genes_file = 'foldseek_combined_results_with_info.pkl'
    if os.path.exists(unique_genes_file):
        # Load the unique genes counts
        counts_unique = read_organelle_counts('organelle_counts_unique_genes.csv')
        
        total_proteins_unique = count_total_proteins(unique_genes_file)
        total_proteins_unique -= 1  # Subtract header row
        
        root_node_unique, all_nodes_unique = build_tree(locations, counts_unique, total_proteins_unique, uniprot_totals)
        
        output_lines_unique = format_tree_output(root_node_unique, is_root=True)
        
        # Save to text file
        with open('organelle_hierarchy_tree_unique_genes.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUBCELLULAR LOCATION HIERARCHY (Unique Genes)\n")
            f.write("="*80 + "\n")
            for line in output_lines_unique:
                f.write(line + "\n")
        
        # Generate and save HTML
        html_lines_unique = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Subcellular Location Hierarchy (Unique Genes)</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
            "h1 { color: #333; }",
            ".search-container { margin-bottom: 20px; position: relative; width: 400px; }",
            ".search-input { width: 100%; padding: 10px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }",
            ".search-input:focus { outline: none; border-color: #0066cc; box-shadow: 0 0 5px rgba(0,102,204,0.3); }",
            ".suggestions { position: absolute; top: 100%; left: 0; width: 100%; background: white; border: 1px solid #ddd; border-top: none; max-height: 200px; overflow-y: auto; display: none; z-index: 1000; }",
            ".suggestion-item { padding: 10px; cursor: pointer; border-bottom: 1px solid #eee; }",
            ".suggestion-item:hover { background-color: #f0f0f0; }",
            ".suggestion-item.highlight { background-color: #e3f2fd; }",
            ".tree-container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
            ".legend { margin: 20px 0; font-size: 12px; }",
            ".legend-item { margin: 5px 0; }",
            ".expand-btn { color: #0066cc; font-size: 12px; }",
            ".term-name { user-select: none; }",
            ".term-name.highlighted { background-color: #ffeb3b; font-weight: bold; }",
            "small { font-size: 11px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Subcellular Location Hierarchy with Protein Localization (Unique Genes)</h1>",
            "<div class='search-container'>",
            "  <input type='text' id='searchInput' class='search-input' placeholder='Search organelles... (e.g., nucleus, mitochondrion)'>",
            "  <div id='suggestions' class='suggestions'></div>",
            "</div>",
            "<div class='legend'>",
            "<h3>Legend:</h3>",
            "<div class='legend-item'>• Count: Number of proteins localized to this term (from organelle_counts_proteins.csv)</div>",
            "<div class='legend-item'>• Percentage: Count as percentage of total proteins analyzed</div>",
            "<div class='legend-item'>• UniProt total: Total human proteins annotated to this term</div>",
            "<div class='legend-item'>• Homologues of UniProt: Count as percentage of UniProt total</div>",
            "<div class='legend-item'>• Primary Parent: Main parent location</div>",
            "<div class='legend-item'>• Secondary Parent(s): Additional parent locations (if term belongs to multiple categories)</div>",
            "<div class='legend-item'>• Click ▶ to expand/collapse child terms (closed by default for easier navigation)</div>",
            "</div>",
            "<div class='tree-container' id='treeContainer'>",
        ]
        
        html_lines_unique.extend(generate_html_tree(root_node_unique))
        
        html_lines_unique.extend([
            "</div>",
            "<p style='font-size: 12px; color: #666; margin-top: 20px;'>",
            f"Total proteins analyzed: {total_proteins_unique}<br>",
            f"Data sources: subcell.txt (UniProt), organelle_counts_proteins.csv, foldseek_results_merged_with_localization_filtered_unique_genes.csv",
            "</p>",
            "",
            "<script>",
            "// Collect all term names for autocomplete",
            "const allTerms = [];",
            "document.querySelectorAll('.term-name').forEach(el => {",
            "  allTerms.push(el.getAttribute('data-term'));",
            "});",
            "",
            "const searchInput = document.getElementById('searchInput');",
            "const suggestionsDiv = document.getElementById('suggestions');",
            "",
            "// Search and suggestion functionality",
            "searchInput.addEventListener('input', function() {",
            "  const query = this.value.toLowerCase().trim();",
            "  ",
            "  if (query.length === 0) {",
            "    suggestionsDiv.style.display = 'none';",
            "    return;",
            "  }",
            "  ",
            "  // Filter suggestions",
            "  const matches = allTerms.filter(term => ",
            "    term.toLowerCase().includes(query)",
            "  ).slice(0, 10);  // Limit to 10 suggestions",
            "  ",
            "  if (matches.length > 0) {",
            "    suggestionsDiv.innerHTML = matches.map((term, idx) => ",
            "      `<div class='suggestion-item' onclick='selectTerm(\"${term}\")' onmouseover='this.classList.add(\"highlight\")' onmouseout='this.classList.remove(\"highlight\")'>${term}</div>`",
            "    ).join('');",
            "    suggestionsDiv.style.display = 'block';",
            "  } else {",
            "    suggestionsDiv.style.display = 'none';",
            "  }",
            "});",
            "",
            "function selectTerm(termName) {",
            "  searchInput.value = termName;",
            "  suggestionsDiv.style.display = 'none';",
            "  highlightAndNavigate(termName);",
            "}",
            "",
            "function highlightAndNavigate(termName) {",
            "  // Remove previous highlighting",
            "  document.querySelectorAll('.term-name').forEach(el => {",
            "    el.classList.remove('highlighted');",
            "  });",
            "  ",
            "  // Find and highlight the matching term",
            "  const termElements = document.querySelectorAll('.term-name');",
            "  let found = false;",
            "  ",
            "  termElements.forEach(el => {",
            "    if (el.getAttribute('data-term').toLowerCase() === termName.toLowerCase()) {",
            "      el.classList.add('highlighted');",
            "      found = true;",
            "      ",
            "      // Expand all parent containers",
            "      let parent = el.closest('div[id^=\"node-\"]');",
            "      while (parent) {",
            "        parent.style.display = 'block';",
            "        const btn = parent.parentElement.querySelector('.expand-btn');",
            "        if (btn) {",
            "          btn.textContent = '▼';",
            "        }",
            "        parent = parent.parentElement.closest('div[id^=\"node-\"]');",
            "      }",
            "      ",
            "      // Scroll to element",
            "      el.scrollIntoView({ behavior: 'smooth', block: 'center' });",
            "    }",
            "  });",
            "  ",
            "  if (!found) {",
            "    alert('Term \"' + termName + '\" not found.');",
            "  }",
            "}",
            "",
            "function toggleExpand(btn, nodeId) {",
            "  const container = document.getElementById('node-' + nodeId);",
            "  if (container.style.display === 'none') {",
            "    container.style.display = 'block';",
            "    btn.textContent = '▼';",
            "  } else {",
            "    container.style.display = 'none';",
            "    btn.textContent = '▶';",
            "  }",
            "}",
            "",
            "// Close suggestions when clicking outside",
            "document.addEventListener('click', function(e) {",
            "  if (e.target !== searchInput) {",
            "    suggestionsDiv.style.display = 'none';",
            "  }",
            "});",
            "</script>",
            "</body>",
            "</html>"
        ])
        
        with open('organelle_hierarchy_tree_unique_genes.html', 'w') as f:
            f.write("\n".join(html_lines_unique))
        
        # Generate statistics dashboard for unique genes
        generate_statistics_dashboard(root_node_unique, all_nodes_unique, total_proteins_unique, 
                                     output_file='organelle_statistics_dashboard_unique_genes.html')

if __name__ == '__main__':
    main()
