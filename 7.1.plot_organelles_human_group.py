# NOTE ON BACKGROUND SIZE (N_total)
# -------------------------------
# In this script, N_total_prot and N_total_gene are computed as the
# **sum of per-compartment counts** for Homo sapiens (excluding any
# 'total' key). This means proteins annotated to multiple compartments
# are **double-counted** in N_total. That makes enrichment tests slightly more
# liberal (p-values a bit smaller) but preserves internal consistency with the
# compartment counts used elsewhere.
#
# TODO: Replace this with a truly deduplicated human universe size
# (number of unique accessions/genes), e.g. by storing a dedicated 'unique_total'
# field when building the pickles and reading that here instead of summing.
#
# HPA → UniProt SL mapping notes
# --------------------------------
# GROUP_DEFS strictly follows the Human Protein Atlas 6-panel / 16-subgroup
# table. HPA terms with no discrete UniProt SL equivalent are skipped:
#   nucleoli rim, actin filaments (generic), aggresome, cytoplasmic bodies,
#   rods & rings, cytokinetic bridge, microtubules (generic),
#   Cilium transition zone, annulus, connecting piece,
#   end piece, equatorial segment.
# Their nearest available relatives are captured by sibling terms in the same
# subgroup (e.g. midbody / spindle for microtubule-based structures).

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from collections import defaultdict
from scipy.stats import hypergeom
import argparse
import io
from PIL import Image
from ciliary_genes import load_ciliary_ensembl_ids

os.system("mkdir -p plots_human_proteins")
os.system("mkdir -p plots_human_genes")

# ── CLI argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--style', choices=['default', 'tight', 'airy'], default='default',
                    help='Visual style preset: default, tight, or airy')
parser.add_argument('--violin-width', type=float, default=None,
                    help='Override violin width (0-1)')
parser.add_argument('--font-scale', type=float, default=None,
                    help='Scale factor for font sizes')
parser.add_argument('--pad-inches', type=float, default=None,
                    help='Padding (inches) when tight-cropping before 16:9 padding')
parser.add_argument('--unify-scale', choices=['none', 'raw', 'percent'], default='none',
                    help='Make protein raw and normalized plots share the same numeric scale (raw counts or percent)')
args, _ = parser.parse_known_args()

if args.style == 'tight':
    VIOLIN_WIDTH = args.violin_width if args.violin_width is not None else 0.55
    FONT_SCALE   = args.font_scale   if args.font_scale   is not None else 0.85
    PAD_INCHES   = args.pad_inches   if args.pad_inches   is not None else 0.01
elif args.style == 'airy':
    VIOLIN_WIDTH = args.violin_width if args.violin_width is not None else 0.95
    FONT_SCALE   = args.font_scale   if args.font_scale   is not None else 1.15
    PAD_INCHES   = args.pad_inches   if args.pad_inches   is not None else 0.05
else:
    VIOLIN_WIDTH = args.violin_width if args.violin_width is not None else 0.75
    FONT_SCALE   = args.font_scale   if args.font_scale   is not None else 1.0
    PAD_INCHES   = args.pad_inches   if args.pad_inches   is not None else 0.02

# ── BH FDR correction ────────────────────────────────────────────────────────

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n     = len(pvals)
    order = np.argsort(pvals)
    q     = pvals[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q             = np.minimum(q, 1.0)
    result        = np.empty(n)
    result[order] = q
    return result


# ── Denominator loader ────────────────────────────────────────────────────────

def _load_denominator(source, kind='proteins'):
    """Load human compartment counts for given source (swissprot/trembl/combined) and kind (proteins/genes)."""
    cache = f'{source}_compartment_counts_{kind}_species.pkl'
    if not os.path.exists(cache):
        print(f'WARNING: {cache} not found.')
        return {}, None
    with open(cache, 'rb') as f:
        data = pickle.load(f)
    human = data.get('homo sapiens', {})
    human_norm = {str(k).strip().lower(): v for k, v in human.items()
                  if str(k).strip().lower() != 'total' and v > 0}
    total = int(sum(human_norm.values()))
    return human_norm, total


# ── Compartment groups, subgroups & display labels ────────────────────────────
# Three-level: Group → Subgroup → [UniProt SL members]
# Mirrors the 6 HPA panels and their 16 sub-category tiles exactly.
# Terms that have no UniProt SL equivalent are noted in comments and skipped.
GROUP_DEFS = [
    # ── 1. Nucleus ────────────────────────────────────────────────────────────
    ('Nucleus', [
        ('Nuclear Membrane',
            ['nucleus membrane']),
        ('Nucleoli',
            ['nucleolus',                       # HPA: Nucleoli
             'nucleolus fibrillar center']),     # HPA: Nucleoli fibrillar center
                                                 # skip: Nucleoli rim (no UniProt SL)
        ('Nucleoplasm',
            ['kinetochore',                      # HPA: Kinetochore
             'chromosome',                       # HPA: Mitotic chromosome
             'nuclear body',                     # HPA: Nuclear bodies
             'nucleus speckle',                  # HPA: Nuclear speckles
             'nucleoplasm']),                    # HPA: Nucleoplasm
    ]),

    # ── 2. Cytoplasm ──────────────────────────────────────────────────────────
    ('Cytoplasm', [
        ('Actin Filaments',
            # skip: Actin filaments (no discrete UniProt SL; cytoskeleton used in IF subgroup)
            ['cleavage furrow',                  # HPA: Cleavage furrow
             'focal adhesion']),                 # HPA: Focal adhesion sites
        ('Centrosome',
            ['centriolar satellite',             # HPA: Centriolar satellite
             'centrosome']),                     # HPA: Centrosome
        ('Cytosol',
            # skip: Aggresome, Cytoplasmic bodies, Rods & rings (no UniProt SL)
            ['cytosol']),                        # HPA: Cytosol
        ('Intermediate Filaments',
            ['cytoskeleton']),                   # HPA: Intermediate filaments (closest UniProt SL)
        ('Microtubules',
            # skip: Cytokinetic bridge (no UniProt SL)
            # skip: Microtubules generic (no UniProt SL separate from cytoskeleton)
            ['spindle pole',                     # HPA: Microtubule ends
             'midbody',                          # HPA: Midbody
             'midbody ring',                     # HPA: Midbody ring
             'spindle']),                        # HPA: Mitotic spindle
        ('Mitochondria',
            ['mitochondrion']),                  # HPA: Mitochondria
    ]),

    # ── 3. Cilium (placed before Endomembrane System so it appears above it) ──
    ('Cilium', [
        ('Cilium',
            ['cilium basal body',                # HPA: Basal body
             'cilium']),                         # HPA: cilium
                                                 # skip: cilium transition zone (no UniProt SL)
    ]),

    # ── 4. Endomembrane System ────────────────────────────────────────────────
    ('Endomembrane System', [
        ('Endoplasmic Reticulum',
            ['endoplasmic reticulum']),           # HPA: Endoplasmic reticulum
        ('Golgi Apparatus',
            ['golgi apparatus']),                 # HPA: Golgi apparatus
        ('Plasma Membrane',
            ['cell junction',                    # HPA: Cell junctions
             'cell membrane']),                  # HPA: Plasma membrane
        ('Vesicles',
            ['endosome',                         # HPA: Endosomes
             'lipid droplet',                    # HPA: Lipid droplets
             'lysosome',                         # HPA: Lysosomes
             'peroxisome',                       # HPA: Peroxisomes
             'vesicle']),                        # HPA: Vesicles
    ]),

    # ── 5. Secretory ─────────────────────────────────────────────────────────
    ('Secretory', [
        ('Secreted Proteins',
            ['secreted']),                       # HPA: Secreted proteins
    ]),

    # ── 6. Sperm ──────────────────────────────────────────────────────────────
    ('Sperm', [
        ('Sperm',
            ['acrosome',                         # HPA: Acrosome
             'calyx',                            # HPA: Calyx
             'centriole']),                      # HPA: Flagellar centriole (basal body of flagellum)
                                                 # skip: Annulus, Connecting piece, End piece,
                                                 #        Equatorial segment (no UniProt SL)
    ]),
]

# Display labels: UniProt SL term → human-readable HPA-style label
DISPLAY = {
    # Nucleus
    'nucleus membrane':           'Nuclear membrane',
    'nucleolus':                  'Nucleoli',
    'nucleolus fibrillar center': 'Nucleoli fibrillar center',
    'kinetochore':                'Kinetochore',
    'chromosome':                 'Mitotic chromosome',
    'nuclear body':               'Nuclear bodies',
    'nucleus speckle':            'Nuclear speckles',
    'nucleoplasm':                'Nucleoplasm',
    # Cytoplasm
    'cleavage furrow':            'Cleavage furrow',
    'focal adhesion':             'Focal adhesion',
    'centriolar satellite':       'Centriolar satellite',
    'centrosome':                 'Centrosome',
    'cytosol':                    'Cytosol',
    'cytoskeleton':               'Intermediate filaments',
    'spindle pole':               'Microtubule ends',
    'midbody':                    'Midbody',
    'midbody ring':               'Midbody ring',
    'spindle':                    'Mitotic spindle',
    'mitochondrion':              'Mitochondria',
    # Endomembrane System
    'endoplasmic reticulum':      'Endoplasmic reticulum',
    'golgi apparatus':            'Golgi apparatus',
    'cell junction':              'Cell junctions',
    'cell membrane':              'Plasma membrane',
    'endosome':                   'Endosomes',
    'lipid droplet':              'Lipid droplets',
    'lysosome':                   'Lysosomes',
    'peroxisome':                 'Peroxisomes',
    'vesicle':                    'Vesicles',
    # Cilium
    'cilium basal body':          'Basal body',
    'cilium':                     'Cilium',
    #'cilium tip':                   'Cilium tip',
    # Secretory
    'secreted':                   'Secreted proteins',
    # Sperm
    'acrosome':                   'Acrosome',
    'calyx':                      'Calyx',
    'centriole':                  'Flagellar centriole',
}

GROUP_COLORS = {
    'Nucleus':              '#e377c2',   # pink
    'Cytoplasm':            '#ff7f0e',   # orange
    'Endomembrane System':  '#1f77b4',   # blue
    'Cilium':               '#2ca02c',   # green
    'Secretory':            '#9467bd',   # purple
    'Sperm':                '#8c564b',   # brown
}

# cilium is now a full member of 'Cilium' above;
# empty string disables the legacy red-bar section.
CILIUM = ''
all_locations = [m for _, sgs in GROUP_DEFS for _, mems in sgs for m in mems]


# ── Layout builder ────────────────────────────────────────────────────────────

def _build_grouped_layout(values_dict, group_defs, group_colors, labels_dict=None):
    """
    Build parallel lists for a grouped, color-coded vertical bar chart.

    Parameters
    ----------
    values_dict : {location: value}
    group_defs  : list of (group_name, [(subgroup_name, [members]), ...])
    group_colors: {group_name: color_str}
    labels_dict : optional display-label overrides for bars

    Returns  (9-tuple)
    -------
    bar_ys, bar_vals, bar_clrs   – data for ax.bar()
    tick_ys, tick_lbls, tick_clrs, levels
        levels element: 'group' | 'subgroup' | 'bar'
    group_spans    : {group_name:             (y_start, y_end, color)}
    subgroup_spans : {(group_name, sg_name):  (y_start, y_end, color)}
    """
    if labels_dict is None:
        labels_dict = {}

    bar_ys, bar_vals, bar_clrs            = [], [], []
    tick_ys, tick_lbls, tick_clrs, levels = [], [], [], []
    group_spans, subgroup_spans           = {}, {}

    y = 0.0
    for g_name, subgroups in group_defs:
        clr = group_colors[g_name]

        # skip entire group when no data present
        if not any(m in values_dict for _, mems in subgroups for m in mems):
            continue

        # group header placeholder (zero-height bar)
        bar_ys.append(y);  bar_vals.append(0.0);  bar_clrs.append('none')
        tick_ys.append(y); tick_lbls.append(g_name); tick_clrs.append(clr)
        levels.append('group')
        y += 1.0

        g_start = y

        for sg_name, members in subgroups:
            present = [(m, values_dict[m]) for m in members if m in values_dict]
            if not present:
                continue
            present.sort(key=lambda x: x[1])

            # subgroup header placeholder (zero-height bar)
            bar_ys.append(y);  bar_vals.append(0.0);  bar_clrs.append('none')
            tick_ys.append(y); tick_lbls.append(sg_name); tick_clrs.append(clr)
            levels.append('subgroup')
            y += 0.65

            sg_start = y
            for m, v in present:
                lbl = labels_dict.get(m) or DISPLAY.get(m) or m.capitalize()
                bar_ys.append(y);  bar_vals.append(v);  bar_clrs.append(clr)
                tick_ys.append(y); tick_lbls.append(lbl); tick_clrs.append(clr)
                levels.append('bar')
                y += 1.0

            subgroup_spans[(g_name, sg_name)] = (sg_start, y - 1.0, clr)
            y += 0.9    # gap between subgroups

        group_spans[g_name] = (g_start, y - 0.9, clr)
        y += 1.6        # gap between groups

    return (bar_ys, bar_vals, bar_clrs,
            tick_ys, tick_lbls, tick_clrs, levels,
            group_spans, subgroup_spans)


# ── Style applicator (violin-file version: font_scale param, axes-fraction label packing,
#    Cilium above Endomembrane System fix) ─────────────────────────────────────

def _apply_grouped_style(ax, tick_ys, tick_lbls, tick_clrs, levels,
                         group_spans, subgroup_spans, font_scale=1.0):
    bar_ticks = [
        (y, lbl, clr)
        for y, lbl, clr, lvl in zip(tick_ys, tick_lbls, tick_clrs, levels)
        if lvl == 'bar'
    ]

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    fig  = ax.get_figure()

    # ── Unit-conversion helpers ───────────────────────────────────────────────
    ax_w_inch = ax.get_position().width  * fig.get_figwidth()
    ax_h_inch = ax.get_position().height * fig.get_figheight()
    xlim      = ax.get_xlim()
    ax_w_data = xlim[1] - xlim[0]
    pts_per_data_x = (72 * ax_w_inch / ax_w_data) if ax_w_data else 1

    # ── Bar labels: rotated text BELOW the x-axis ────────────────────────────
    if bar_ticks:
        ys, lbls, clrs = zip(*bar_ticks)
        ax.set_xticks(list(ys))
        ax.set_xticklabels(['']*len(ys))
        trans = ax.get_xaxis_transform()
        for y_bar, lbl, clr in zip(ys, lbls, clrs):
            ax.text(y_bar, -0.02, lbl,
                    transform=trans,
                    ha='right', va='top', fontsize=8.5 * font_scale, color=clr,
                    rotation=45, rotation_mode='anchor',
                    clip_on=False)
    else:
        ax.set_xticks([])

    # Place subgroup labels using axes-fraction vertical coordinates so their
    # placement is stable under linear or log y-scales.
    sg_fontsize   = 10.5 * font_scale
    char_w_pts    = sg_fontsize * 0.60
    move_labels_out = getattr(ax, 'move_labels_out', False)

    # Points per axes fraction (vertical): 1.0 axes fraction == ax_h_inch inches == 72 * ax_h_inch points
    pts_per_axes_y = 72 * ax_h_inch

    # Base y position in axes fraction units (just above the top of the axes)
    if move_labels_out:
        base_sg_ax = 1.18
    else:
        base_sg_ax = 1.02

    # compute label height in axes fraction units and ensure lane spacing is at least that
    label_height_ax = (sg_fontsize * 1.2) / pts_per_axes_y if pts_per_axes_y else sg_fontsize * 0.02
    # increase subgroup lane spacing multiplier to reduce vertical crowding
    sg_step_ax = max((sg_fontsize * 1.6) / pts_per_axes_y if pts_per_axes_y else sg_fontsize * 0.02,
                     label_height_ax * 1.6)
    gap_ax = 0.01

    lane_end_x = []
    lane_for_sg = {}

    trans_xaxis = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    ax_pos = ax.get_position()

    for (g_name, sg_name), (y0, y1, clr) in sorted(
            subgroup_spans.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / 2):
        ax.axvspan(y0 - 0.45, y1 + 0.45, alpha=0.13, color=clr, zorder=0)
        ax.axvline(y0 - 0.5, color=clr, linewidth=0.8, alpha=0.4, zorder=1)
        ax.axvline(y1 + 0.5, color=clr, linewidth=0.8, alpha=0.4, zorder=1)

        x_ctr = (y0 + y1) / 2

        # Convert x center to axes-fraction horizontal coordinate (robust to log scales)
        disp = ax.transData.transform((x_ctr, 0.0))
        fig_frac = fig.transFigure.inverted().transform(disp)
        fig_x = fig_frac[0]
        x_ax_frac = (fig_x - ax_pos.x0) / ax_pos.width if ax_pos.width else 0.5

        # compute label width in axes fraction units from font metrics
        label_width_ax = (len(sg_name) * char_w_pts) / (72.0 * ax_w_inch) if ax_w_inch else 0.05
        x_lbl_start_ax = x_ax_frac - label_width_ax / 2.0
        x_lbl_end_ax   = x_ax_frac + label_width_ax / 2.0

        assigned = False
        for li, end_x_ax in enumerate(lane_end_x):
            if x_lbl_start_ax >= end_x_ax + gap_ax:
                lane_end_x[li] = x_lbl_end_ax
                lane_for_sg[(g_name, sg_name)] = li
                assigned = True
                break
        if not assigned:
            lane_for_sg[(g_name, sg_name)] = len(lane_end_x)
            lane_end_x.append(x_lbl_end_ax)

        li       = lane_for_sg[(g_name, sg_name)]
        y_txt_ax = base_sg_ax + li * sg_step_ax
        ax.text(x_ctr, y_txt_ax, sg_name,
                transform=trans_xaxis,
                ha='center', va='bottom', fontsize=sg_fontsize, color=clr,
                fontstyle='italic', zorder=10, clip_on=False)

    n_sg_lanes = max(len(lane_end_x), 1)

    grp_fontsize = 13 * font_scale
    # Compute base y for group labels in axes fraction units (just above subgroup lanes)
    if move_labels_out:
        base_group_y_ax = base_sg_ax + n_sg_lanes * sg_step_ax + 0.08
    else:
        base_group_y_ax = base_sg_ax + n_sg_lanes * sg_step_ax + 0.01

    # Determine horizontal packing lanes for group labels to avoid overlaps
    group_lane_end_x = []
    group_for_g = {}
    # convert group step to axes fraction units based on font size
    group_step_ax = (grp_fontsize * 2.0 / pts_per_axes_y) if pts_per_axes_y else grp_fontsize * 0.02
    # ensure group_step is at least label height
    label_height_group_ax = (grp_fontsize * 1.2) / pts_per_axes_y if pts_per_axes_y else grp_fontsize * 0.02
    group_step_ax = max(group_step_ax, label_height_group_ax * 1.6)

    # First pass: determine horizontal packing lanes for group labels and record entries
    group_entries = []
    for g_name, (y0, y1, clr) in sorted(group_spans.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / 2):
        ax.axvspan(y0 - 0.45, y1 + 0.45, alpha=0.05, color=clr, zorder=0)
        x_ctr = (y0 + y1) / 2

        # Convert x center to axes-fraction horizontal coordinate (robust to log scales)
        disp = ax.transData.transform((x_ctr, 0.0))
        fig_frac = fig.transFigure.inverted().transform(disp)
        fig_x = fig_frac[0]
        x_ax_frac = (fig_x - ax_pos.x0) / ax_pos.width if ax_pos.width else 0.5

        half_w_ax      = (len(g_name) * char_w_pts / 2.0) / (72.0 * ax_w_inch) if ax_w_inch else 0.05
        x_lbl_start_ax = x_ax_frac - half_w_ax
        x_lbl_end_ax   = x_ax_frac + half_w_ax

        assigned = False
        for li, end_x_ax in enumerate(group_lane_end_x):
            if x_lbl_start_ax >= end_x_ax + gap_ax:
                group_lane_end_x[li] = x_lbl_end_ax
                group_for_g[g_name] = li
                assigned = True
                break
        if not assigned:
            group_for_g[g_name] = len(group_lane_end_x)
            group_lane_end_x.append(x_lbl_end_ax)

        group_entries.append((g_name, x_ctr, clr))

    # Compute y positions for each group label (axes fraction units)
    group_label_y_ax = {}
    for g_name, li in group_for_g.items():
        group_label_y_ax[g_name] = base_group_y_ax + li * group_step_ax

    # Ensure Cilium label appears above Endomembrane System and avoid overlapping
    if 'Cilium' in group_label_y_ax and 'Endomembrane System' in group_label_y_ax:
        if group_label_y_ax['Cilium'] <= group_label_y_ax['Endomembrane System']:
            max_lane = max(group_for_g.values()) if group_for_g else 0
            # Move Cilium to a new top lane
            group_for_g['Cilium'] = max_lane + 1
            group_label_y_ax['Cilium'] = base_group_y_ax + (max_lane + 1) * group_step_ax

    # Second pass: actually draw group labels at computed y positions
    for g_name, x_ctr, clr in group_entries:
        y_txt_ax = group_label_y_ax.get(g_name, base_group_y_ax)
        ax.text(x_ctr, y_txt_ax, g_name,
                transform=trans_xaxis,
                ha='center', va='bottom', fontsize=grp_fontsize, fontweight='bold',
                color=clr, zorder=11, clip_on=False)

    ax.set_ylim(ylim)


# ── Helpers to compute per-query counts from foldseek results ────────────────

def _select_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_per_query_counts(df, ciliary_set, tree, mode='gene'):
    """Return (query_list, per_query_counts) where per_query_counts[query][org]=count.

    mode: 'gene' => deduplicate by (target_gene_name, species)
          'protein' => deduplicate by target_uniprot_id
    """
    query_col        = _select_column(df, ['query_ensembl_id', 'query_uniprot_id', 'query_gene_name'])
    species_col      = _select_column(df, ['species', 'organism', 'taxon_id'])
    target_gene_col  = _select_column(df, ['target_gene_name', 'target_protein_name'])
    target_uniprot_col = _select_column(df, ['target_uniprot_id'])
    loc_col          = _select_column(df, ['target_localization_ancestors', 'target_localization_uniprot'])

    if query_col is None:
        raise RuntimeError('query column not found in foldseek DataFrame')
    if loc_col is None:
        raise RuntimeError('target localization column not found in foldseek DataFrame')

    present_queries = set(df[query_col].dropna().astype(str).unique())
    queries = sorted([q for q in ciliary_set if str(q) in present_queries])
    print(f'Found {len(queries):,} ciliary queries present in foldseek results (mode={mode})')

    per_query = {q: defaultdict(int) for q in queries}

    grouped = df.groupby(query_col)
    for q in queries:
        try:
            g = grouped.get_group(q)
        except KeyError:
            continue
        seen = set()
        for _, row in g.iterrows():
            if mode == 'gene':
                tg = row.get(target_gene_col) if target_gene_col else None
                sp = row.get(species_col)     if species_col     else None
                if pd.isna(tg) or pd.isna(sp):
                    key = (row.get(target_uniprot_col), )
                else:
                    key = (str(tg).strip(), str(sp).strip())
            else:
                key = (row.get(target_uniprot_col), )

            if not key or any(pd.isna(x) for x in key):
                continue
            if key in seen:
                continue
            seen.add(key)

            locs = row.get(loc_col) or []
            if isinstance(locs, str):
                try:
                    import ast
                    locs = ast.literal_eval(locs)
                except Exception:
                    locs = [locs]
            if not isinstance(locs, (list, tuple, set)):
                locs = [locs]

            for loc in locs:
                if not loc:
                    continue
                loc_k = str(loc).strip().lower()
                if loc_k not in tree:
                    continue
                for ancestor in tree[loc_k]:
                    per_query[q][ancestor] += 1

    return queries, per_query


def get_grouped_order(values_dict, group_defs):
    order = []
    for g_name, subgroups in group_defs:
        if not any(m in values_dict for _, mems in subgroups for m in mems):
            continue
        for sg_name, members in subgroups:
            present = [(m, values_dict[m]) for m in members if m in values_dict]
            if not present:
                continue
            present.sort(key=lambda x: x[1])
            for m, v in present:
                order.append(m)
    return order


def make_dataset_for_orgs(orgs, queries, per_query_counts):
    dataset = []
    for o in orgs:
        vals = []
        for q in queries:
            v = per_query_counts.get(q, {}).get(o, 0)
            try:
                if pd.isna(v):
                    vals.append(0.0)
                else:
                    vals.append(float(v))
            except Exception:
                vals.append(0.0)
        dataset.append(np.array(vals, dtype=float))
    return dataset


# ── Violin plotting helper ────────────────────────────────────────────────────

def plot_grouped_violins(fig, ax, values_dict_for_ordering, queries, per_query_counts,
                         ylabel, title, outpath, labels_dict=None, normalize_by=None,
                         percent_scale=False, violin_width=None, font_scale=None, pad_inches=None,
                         dataset_override=None, force_ylim=None, yscale=None):
    # Resolve visual parameters
    violin_width = violin_width if violin_width is not None else VIOLIN_WIDTH
    font_scale   = font_scale   if font_scale   is not None else FONT_SCALE
    pad_inches   = pad_inches   if pad_inches   is not None else PAD_INCHES

    _bys, _bvs, _bcs, _tys, _tls, _tcs, _lvls, _spans, _subspans = _build_grouped_layout(
        values_dict_for_ordering, GROUP_DEFS, GROUP_COLORS, labels_dict=labels_dict or {}
    )

    bar_positions = [y for y, lvl in zip(_bys, _lvls) if lvl == 'bar']
    org_order     = get_grouped_order(values_dict_for_ordering, GROUP_DEFS)
    bar_colors    = [c for c, lvl in zip(_bcs, _lvls) if lvl == 'bar']

    if dataset_override is not None:
        dataset = dataset_override
    else:
        dataset = make_dataset_for_orgs(org_order, queries, per_query_counts)

    if normalize_by is not None:
        lower_denom = {str(k).strip().lower(): v for k, v in normalize_by.items()}
        normed = []
        for arr, o in zip(dataset, org_order):
            denom = lower_denom.get(str(o).strip().lower(), 0)
            if denom:
                normed.append(arr / float(denom))
            else:
                normed.append(np.zeros_like(arr))
        dataset = normed

    if percent_scale:
        dataset = [arr * 100.0 for arr in dataset]

    if yscale == 'log':
        sanitized = []
        for arr in dataset:
            try:
                arr_np  = np.asarray(arr, dtype=float)
                mask    = np.isfinite(arr_np) & (arr_np > 0)
                arr_pos = arr_np[mask]
                sanitized.append(np.array(arr_pos, dtype=float))
            except Exception:
                sanitized.append(np.array([], dtype=float))
        dataset = sanitized

    parts = ax.violinplot(dataset, positions=bar_positions, widths=violin_width,
                          showmeans=False, showmedians=True)
    for body, clr in zip(parts['bodies'], bar_colors):
        body.set_facecolor(clr)
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    if 'cmedians' in parts and parts['cmedians'] is not None:
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.0)

    if force_ylim is not None:
        if yscale == 'log':
            low, high = force_ylim
            if low <= 0:
                min_positive = None
                for arr in dataset:
                    try:
                        arrp = np.asarray(arr, dtype=float)
                        arrp = arrp[np.isfinite(arrp) & (arrp > 0)]
                        if arrp.size > 0:
                            v = float(np.nanmin(arrp))
                            if min_positive is None or v < min_positive:
                                min_positive = v
                    except Exception:
                        continue
                low = min_positive if min_positive is not None else 1.0
            if high <= low:
                high = low * 10.0
            ax.set_ylim((low, high))
        else:
            ax.set_ylim(force_ylim)
    elif percent_scale:
        max_val = 0.0
        for arr in dataset:
            try:
                if len(arr) > 0:
                    max_val = max(max_val, float(np.nanmax(arr)))
            except Exception:
                continue
        if max_val <= 0:
            top = 1.0
        else:
            top = max(5.0, max_val * 1.10)
        ax.set_ylim(0, top)
        if top >= 20:
            fmt = '%.0f%%'
        else:
            fmt = '%.1f%%'
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))

    if yscale == 'log':
        ax.set_yscale('log')

    _apply_grouped_style(ax, _tys, _tls, _tcs, _lvls, _spans, _subspans, font_scale=font_scale)

    ax.set_ylabel(ylabel)
    if title:
        fig = ax.get_figure()
        fig.text(0.995, 0.5, title, fontsize=14 * font_scale, fontweight='bold',
                 ha='left', va='center', rotation=90)

    ax.grid(axis='y', alpha=0.3)
    fig = ax.get_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=pad_inches)
    buf.seek(0)
    im = Image.open(buf).convert('RGBA')
    w_tight, h_tight = im.size

    target_w = max(w_tight, int(np.ceil(h_tight * 16.0 / 9.0)))
    target_h = int(np.ceil(target_w * 9.0 / 16.0))
    if target_h < h_tight:
        target_h = h_tight
        target_w = int(np.ceil(target_h * 16.0 / 9.0))

    bg    = Image.new('RGBA', (target_w, target_h), (255, 255, 255, 255))
    off_x = (target_w - w_tight) // 2
    off_y = (target_h - h_tight) // 2
    bg.paste(im, (off_x, off_y), im)
    out_img = bg.convert('RGB')
    out_img.save(outpath, format='PNG', dpi=(300, 300))
    print(f'Plot saved as {outpath}')


def _presentation_figsize(n_items):
    """Return a presentation-friendly (width, height) for n items."""
    min_width      = 10.0
    max_width      = 20.0
    scale_per_item = 0.60
    width  = min(max_width, max(min_width, n_items * scale_per_item))
    height = width * 9.0 / 16.0
    return (width, height)


# ── Enrichment plots function ─────────────────────────────────────────────────

def _enrichment_plots(k_vals, K_vals, organelles, out_dir, label, N_total, n_total,
                      denom_source='swissprot'):
    """Bar plots of -log10(FDR q-value) per compartment, grouped by category."""
    N = int(N_total) if N_total > 0 else 1
    n = int(n_total) if n_total > 0 else 1
    print(f'[{label}] N={N:,}  n={n:,}')

    pvals = np.array([
        hypergeom.sf(k - 1, N, K, n) if K > 0 and k > 0 else 1.0
        for k, K in zip(k_vals, K_vals)
    ])
    qvals    = bh_fdr(pvals)
    neg_logq = -np.log10(np.maximum(qvals, 1e-300))
    sig_line = -np.log10(0.01)

    k_dict = dict(zip(organelles, k_vals))
    K_dict = dict(zip(organelles, K_vals))

    # ── all compartments ──────────────────────────────────────────────────────
    nlq_dict  = dict(zip(organelles, neg_logq))
    _lbls_e   = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(k_dict.get(o,0))}/{int(K_dict.get(o,0))})"
                 for o in nlq_dict}
    _be_ys, _be_vs, _be_cs, _te_ys, _te_ls, _te_cs, _te_lvls, _te_spans, _te_subspans = \
        _build_grouped_layout(nlq_dict, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls_e)
    fig_e, ax_e = plt.subplots(figsize=(max(10, len(_be_ys) * 0.45), 14))
    ax_e.bar(_be_ys, _be_vs, color=_be_cs, width=0.75)
    _apply_grouped_style(ax_e, _te_ys, _te_ls, _te_cs, _te_lvls, _te_spans, _te_subspans)
    ax_e.axhline(sig_line, color='red', linewidth=1.2, linestyle='--', alpha=0.7,
                 label='FDR q < 0.01')
    ax_e.set_ylabel('−log₁₀(FDR q-value)')
    ax_e.text(1.02, 0.5, f'Ciliary Homologues: Hypergeometric Enrichment by Compartment ({label})',
              fontsize=16, fontweight='bold', rotation=-90, va='center', ha='left',
              transform=ax_e.transAxes)
    ax_e.legend(loc='upper right', fontsize=11)
    ax_e.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, f'organelle_hypergeom_enrichment_{label.lower().replace(" ", "_")}_{denom_source}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {out}")
    plt.close(fig_e)

    # ── significant only ──────────────────────────────────────────────────────
    sig_mask = qvals < 0.01
    if sig_mask.sum() > 0:
        sig_orgs = np.array(organelles)[sig_mask]
        sig_nlq  = neg_logq[sig_mask]
        sig_k    = k_vals[sig_mask]
        sig_K    = K_vals[sig_mask]
        sig_dict = dict(zip(sig_orgs, sig_nlq))
        _lbls_es = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(k)}/{int(K)})"
                    for o, k, K in zip(sig_orgs, sig_k, sig_K)}
        _bes_ys, _bes_vs, _bes_cs, _tes_ys, _tes_ls, _tes_cs, _tes_lvls, _tes_spans, _tes_subspans = \
            _build_grouped_layout(sig_dict, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls_es)
        fig_es, ax_es = plt.subplots(figsize=(max(10, len(_bes_ys) * 0.45), 14))
        ax_es.bar(_bes_ys, _bes_vs, color=_bes_cs, width=0.75)
        _apply_grouped_style(ax_es, _tes_ys, _tes_ls, _tes_cs, _tes_lvls, _tes_spans, _tes_subspans)
        ax_es.axhline(sig_line, color='red', linewidth=1.2, linestyle='--', alpha=0.7,
                      label='FDR q < 0.01')
        ax_es.set_ylabel('−log₁₀(FDR q-value)')
        ax_es.text(1.02, 0.5,
                   f'Ciliary Homologues: Significant Hypergeometric Enrichment (FDR<0.01) — {label}',
                   fontsize=16, fontweight='bold', rotation=-90, va='center', ha='left',
                   transform=ax_es.transAxes)
        ax_es.legend(loc='upper right', fontsize=11)
        ax_es.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(out_dir,
                            f'organelle_hypergeom_enrichment_{label.lower().replace(" ", "_")}_{denom_source}_sig.png')
        plt.savefig(out2, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {out2}")
        plt.close(fig_es)
    else:
        print(f"[{label}] No compartments significant at FDR < 0.01")


# ── Load hierarchy ────────────────────────────────────────────────────────────
with open('subcell_hierarchy.pkl', 'rb') as f:
    tree = pickle.load(f)

root_locations = [key for key, value in tree.items() if isinstance(value, list) and value == [key]]
print(f"Found {len(root_locations)} root subcellular locations")


# ── Load protein numerator ────────────────────────────────────────────────────
df_prot = pd.read_csv("organelle_counts_proteins.csv")
df_prot = df_prot[df_prot['Organelle'].isin(all_locations)]
prot_col = None
for col in df_prot.columns:
    if col.strip().lower() == 'homo sapiens':
        prot_col = col
        break
if prot_col is None:
    raise ValueError("Column for 'homo sapiens' not found in organelle_counts_proteins.csv")
prot_counts   = df_prot.set_index('Organelle')[prot_col]
organelle_col = prot_counts.index.to_series()
human_counts  = pd.to_numeric(prot_counts, errors='coerce').fillna(0).to_numpy(dtype=float)

print(f"Processing Homo sapiens only, {len(organelle_col)} compartments...")

sort_idx      = np.argsort(human_counts)[::-1]
organelle_col = organelle_col.iloc[sort_idx]
human_counts  = human_counts[sort_idx]

n_total_prot  = int(human_counts.sum())

prot_values_dict = dict(zip(organelle_col, human_counts))


# ── Load gene numerator ───────────────────────────────────────────────────────
try:
    df_g = pd.read_csv("organelle_counts_genes.csv")
    df_g = df_g[df_g['Organelle'].isin(all_locations)]
    gene_col = None
    for col in df_g.columns:
        if col.strip().lower() == 'homo sapiens':
            gene_col = col
            break
    if gene_col is None:
        raise ValueError("Column for 'homo sapiens' not found in organelle_counts_genes.csv")
    gene_counts     = df_g.set_index('Organelle')[gene_col]
    organelle_col_g = gene_counts.index.to_series()
    human_counts_g  = pd.to_numeric(gene_counts, errors='coerce').fillna(0).to_numpy(dtype=float)
    sort_idx_g      = np.argsort(human_counts_g)[::-1]
    organelle_col_g = organelle_col_g.iloc[sort_idx_g]
    human_counts_g  = human_counts_g[sort_idx_g]
    n_total_gene    = int(human_counts_g.sum())
    gene_values_dict = dict(zip(organelle_col_g, human_counts_g))
except Exception as e:
    print(f'Warning: failed to load gene aggregated counts: {e}')
    gene_values_dict = {}
    organelle_col_g  = pd.Series([], dtype=str)
    human_counts_g   = np.array([])
    n_total_gene     = 0


# ── Load per-query caches (produced by 7.2.count_per_query.py) ───────────────
gene_cache = 'per_query_organelle_counts_genes.csv'
prot_cache = 'per_query_organelle_counts_proteins.csv'

ciliary_ids = load_ciliary_ensembl_ids()
ciliary_set = set(ciliary_ids)

per_query_gene_counts = None
per_query_prot_counts = None
queries_gene = []
queries_prot = []


def _load_per_query_csv(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    queries = [str(q) for q in df.index.astype(str)]
    perq = {}
    for q in queries:
        row = df.loc[q].to_dict()
        d = {}
        for col in df.columns:
            if col == 'Total':
                continue
            if col in all_locations:
                value = row.get(col)
                try:
                    d[col] = int(value) if value is not None and not pd.isna(value) else 0
                except Exception:
                    try:
                        d[col] = int(float(value)) if value is not None and not pd.isna(value) else 0
                    except Exception:
                        d[col] = 0
        perq[q] = d
    return queries, perq


def _load_per_query_fraction_csv(path):
    """Load a per-query fraction CSV (values in [0,1]) and return queries, per_query dict."""
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    queries = [str(q) for q in df.index.astype(str)]
    perq = {}
    for q in queries:
        row = df.loc[q].to_dict()
        d = {}
        for col in df.columns:
            if col == 'Total':
                continue
            if col in all_locations:
                value = row.get(col)
                try:
                    d[col] = float(value) if value is not None and not pd.isna(value) else 0.0
                except Exception:
                    try:
                        d[col] = float(str(value).strip()) if value is not None and not pd.isna(value) else 0.0
                    except Exception:
                        d[col] = 0.0
        perq[q] = d
    return queries, perq


if os.path.exists(gene_cache):
    print(f'Loading per-query gene cache from {gene_cache} ...')
    queries_gene, per_query_gene_counts = _load_per_query_csv(gene_cache)

if os.path.exists(prot_cache):
    print(f'Loading per-query protein cache from {prot_cache} ...')
    queries_prot, per_query_prot_counts = _load_per_query_csv(prot_cache)

# If either cache is missing, compute the missing one(s) from the foldseek pickle
if per_query_gene_counts is None or per_query_prot_counts is None:
    if not os.path.exists('foldseek_combined_results_with_info.pkl'):
        raise FileNotFoundError('foldseek_combined_results_with_info.pkl not found; required to compute missing per-query caches.')
    print('Loading foldseek combined results (this may be large) ...')
    df_fold = pd.read_pickle('foldseek_combined_results_with_info.pkl')
    print(f'  Loaded {len(df_fold):,} rows')

    if per_query_gene_counts is None:
        print('Building per-query gene-level counts (deduplicated by gene+species)')
        queries_gene, per_query_gene_counts = build_per_query_counts(df_fold, ciliary_set, tree, mode='gene')

    if per_query_prot_counts is None:
        print('Building per-query protein-level counts (deduplicated by UniProt accession)')
        queries_prot, per_query_prot_counts = build_per_query_counts(df_fold, ciliary_set, tree, mode='protein')

# Ensure lists
queries_gene = list(queries_gene)
queries_prot = list(queries_prot)

print(f"Protein enrichment: n={n_total_prot:,} (duplicated)")
print(f"Gene enrichment: n={n_total_gene:,} (duplicated)")


# ============================================================================
# ALL PLOTS (loop over denom_source)
# ============================================================================

for denom_source in ('swissprot', 'trembl', 'combined'):
    print(f"\n\n===== Normalized plots: denom_source={denom_source} =====")

    # Load denominators for this source
    prot_denom_dict, N_total_prot = _load_denominator(denom_source, 'proteins')
    gene_denom_dict, N_total_gene = _load_denominator(denom_source, 'genes')

    if N_total_prot is None:
        print(f'  WARNING: protein denominator totals not available for {denom_source}; skipping protein normalized plots.')
    if N_total_gene is None:
        print(f'  WARNING: gene denominator totals not available for {denom_source}; skipping gene normalized plots.')

    # Load per-query fraction caches for this denom_source
    gene_frac_cache = f'per_query_organelle_counts_genes_{denom_source}_protein_denominator_fraction.csv'
    prot_frac_cache = f'per_query_organelle_counts_proteins_{denom_source}_protein_denominator_fraction.csv'

    per_query_gene_counts_fraction = None
    per_query_prot_counts_fraction = None
    queries_gene_fraction = []
    queries_prot_fraction = []

    if os.path.exists(gene_frac_cache):
        print(f'Loading per-query gene fraction cache from {gene_frac_cache} ...')
        queries_gene_fraction, per_query_gene_counts_fraction = _load_per_query_fraction_csv(gene_frac_cache)

    if os.path.exists(prot_frac_cache):
        print(f'Loading per-query protein fraction cache from {prot_frac_cache} ...')
        queries_prot_fraction, per_query_prot_counts_fraction = _load_per_query_fraction_csv(prot_frac_cache)

    # ── Protein normalized plots ─────────────────────────────────────────────

    if N_total_prot is not None:
        sp_denom  = np.array([prot_denom_dict.get(o, 0) for o in organelle_col], dtype=float)
        sp_fracs  = np.where(sp_denom > 0, human_counts / sp_denom, 0.0)

        # ── Plot 3: Normalized by denominator total (all compartments) ────────
        print(f"\nCreating protein normalized plot ({denom_source}, all compartments)...")
        _sp_dict  = dict(zip(organelle_col, sp_fracs))
        _raw_dict = dict(zip(organelle_col, human_counts))
        _den_dict = dict(zip(organelle_col, sp_denom))
        _lbls3 = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(_raw_dict[o])}/{int(_den_dict[o])})"
                  for o in _sp_dict}
        _bys3, _bvs3, _bcs3, _tys3, _tls3, _tcs3, _lvls3, _spans3, _subspans3 = \
            _build_grouped_layout(_sp_dict, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls3)
        fig3, ax3 = plt.subplots(figsize=(max(10, len(_bys3) * 0.45), 14))
        ax3.bar(_bys3, _bvs3, color=_bcs3, width=0.75)
        setattr(ax3, 'move_labels_out', True)
        _apply_grouped_style(ax3, _tys3, _tls3, _tcs3, _lvls3, _spans3, _subspans3)
        ax3.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax3.set_ylabel(f'Fraction of {denom_source} compartment total (%)')
        ax3.text(1.02, 0.5,
                 f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (Human)',
                 fontsize=16, fontweight='bold', rotation=-90, va='center', ha='left',
                 transform=ax3.transAxes)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1)
        yticks = np.linspace(0, 1, 11)
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([f'{int(x*100)}' for x in yticks])
        plt.tight_layout()
        plt.savefig(f'plots_human_proteins/organelle_human_fraction_{denom_source}.png',
                    dpi=300, bbox_inches='tight')
        print(f"Plot saved as plots_human_proteins/organelle_human_fraction_{denom_source}.png")
        plt.close(fig3)

        # ── Plot 4: Normalized filtered (fraction >= 0.05) ────────────────────
        print(f"\nCreating protein normalized plot filtered ({denom_source}, fraction >= 0.05)...")
        mask_05 = sp_fracs >= 0.05
        print(f"Filtered to {mask_05.sum()} compartments with normalized fraction >= 0.05")
        if mask_05.sum() > 0:
            _orgs_f  = organelle_col[mask_05]
            _fracs_f = sp_fracs[mask_05]
            _raw_f   = human_counts[mask_05]
            _den_f   = sp_denom[mask_05]
            _sp_dict4 = dict(zip(_orgs_f, _fracs_f))
            _lbls4 = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(r)}/{int(d)})"
                      for o, r, d in zip(_orgs_f, _raw_f, _den_f)}
            _bys4, _bvs4, _bcs4, _tys4, _tls4, _tcs4, _lvls4, _spans4, _subspans4 = \
                _build_grouped_layout(_sp_dict4, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls4)
            fig4, ax4 = plt.subplots(figsize=(max(10, len(_bys4) * 0.45), 14))
            ax4.bar(_bys4, _bvs4, color=_bcs4, width=0.75)
            setattr(ax4, 'move_labels_out', True)
            _apply_grouped_style(ax4, _tys4, _tls4, _tcs4, _lvls4, _spans4, _subspans4)
            ax4.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
            ax4.set_ylabel(f'Fraction of {denom_source} compartment total (%)')
            ax4.text(1.02, 0.5,
                     f'Ciliary Homologues: Proteins Normalized >= 5% of {denom_source.title()} Compartment Total (Human)',
                     fontsize=16, fontweight='bold', rotation=-90, va='center', ha='left',
                     transform=ax4.transAxes)
            ax4.grid(axis='y', alpha=0.3)
            ax4.set_ylim(0, 1)
            yticks = np.linspace(0, 1, 11)
            ax4.set_yticks(yticks)
            ax4.set_yticklabels([f'{int(x*100)}' for x in yticks])
            plt.tight_layout()
            plt.savefig(f'plots_human_proteins/organelle_human_fraction_filtered_{denom_source}.png',
                        dpi=300, bbox_inches='tight')
            print(f"Plot saved as plots_human_proteins/organelle_human_fraction_filtered_{denom_source}.png")
            plt.close(fig4)
        else:
            print("No compartments passed the >= 0.05 threshold")

        # ── Violin 3: Normalized protein fraction (all compartments) ─────────
        if prot_values_dict and per_query_prot_counts is not None:
            out_v3 = f'plots_human_proteins/organelle_human_fraction_violin_{denom_source}.png'
            fig_v3, ax_v3 = plt.subplots(figsize=_presentation_figsize(len(prot_values_dict)))
            setattr(ax_v3, 'move_labels_out', True)

            # Determine unify-scale settings
            unify = getattr(args, 'unify_scale', 'none')
            org_order_v3 = get_grouped_order(prot_values_dict, GROUP_DEFS)
            raw_dataset_v3 = make_dataset_for_orgs(org_order_v3, queries_prot, per_query_prot_counts)

            force_ylim_vals = None
            norm_counts_dataset = None
            norm_pct_dataset    = None

            if unify in ('raw', 'percent'):
                if per_query_prot_counts_fraction is not None:
                    aligned_prot_frac_u = {q: per_query_prot_counts_fraction.get(q, {}) for q in queries_prot}
                    norm_frac_dataset = make_dataset_for_orgs(org_order_v3, queries_prot, aligned_prot_frac_u)
                else:
                    norm_frac_dataset = []
                    lower_denom_u = {str(k).strip().lower(): v for k, v in prot_denom_dict.items()}
                    for arr, o in zip(raw_dataset_v3, org_order_v3):
                        denom = lower_denom_u.get(str(o).strip().lower(), 0)
                        if denom:
                            norm_frac_dataset.append(arr / float(denom))
                        else:
                            norm_frac_dataset.append(np.zeros_like(arr))

                if unify == 'raw':
                    lower_denom_u = {str(k).strip().lower(): v for k, v in prot_denom_dict.items()}
                    norm_counts_dataset = []
                    for arr, o in zip(norm_frac_dataset, org_order_v3):
                        denom = lower_denom_u.get(str(o).strip().lower(), 0)
                        norm_counts_dataset.append(arr * float(denom) if denom else np.zeros_like(arr))
                    max_val = 0.0
                    for ds in (raw_dataset_v3, norm_counts_dataset):
                        for arr in ds:
                            try:
                                if len(arr) > 0:
                                    max_val = max(max_val, float(np.nanmax(arr)))
                            except Exception:
                                continue
                    top = max(5.0, max_val * 1.10) if max_val > 0 else 1.0
                    force_ylim_vals = (0, top)

                elif unify == 'percent':
                    lower_denom_u = {str(k).strip().lower(): v for k, v in prot_denom_dict.items()}
                    raw_pct_dataset_u  = []
                    norm_pct_dataset   = []
                    for raw_arr, frac_arr, o in zip(raw_dataset_v3, norm_frac_dataset, org_order_v3):
                        denom = lower_denom_u.get(str(o).strip().lower(), 0)
                        if denom:
                            raw_pct_dataset_u.append((raw_arr / float(denom)) * 100.0)
                            norm_pct_dataset.append(frac_arr * 100.0)
                        else:
                            raw_pct_dataset_u.append(np.zeros_like(raw_arr))
                            norm_pct_dataset.append(np.zeros_like(frac_arr))
                    max_val = 0.0
                    for ds in (raw_pct_dataset_u, norm_pct_dataset):
                        for arr in ds:
                            try:
                                if len(arr) > 0:
                                    max_val = max(max_val, float(np.nanmax(arr)))
                            except Exception:
                                continue
                    top = max(5.0, max_val * 1.10) if max_val > 0 else 1.0
                    force_ylim_vals = (0, top)

            if per_query_prot_counts_fraction is not None:
                aligned_prot_frac = {q: per_query_prot_counts_fraction.get(q, {}) for q in queries_prot}
                if unify == 'raw' and norm_counts_dataset is not None:
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, aligned_prot_frac,
                                 ylabel='Homologues (unique target proteins) per ciliary gene',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene, shown as counts)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES,
                                 dataset_override=norm_counts_dataset, force_ylim=force_ylim_vals)
                elif unify == 'percent' and norm_pct_dataset is not None:
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, aligned_prot_frac,
                                 ylabel='Percentage of compartment total (per ciliary gene)',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES,
                                 dataset_override=norm_pct_dataset, force_ylim=force_ylim_vals)
                else:
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, aligned_prot_frac,
                                 ylabel='Percentage of compartment total (per ciliary gene)',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 percent_scale=True,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES)
            else:
                # fallback: normalize on the fly using denominators
                if unify == 'raw' and force_ylim_vals is not None:
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, per_query_prot_counts,
                                 ylabel='Homologues (unique target proteins) per ciliary gene',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene, shown as counts)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES,
                                 dataset_override=raw_dataset_v3, force_ylim=force_ylim_vals)
                elif unify == 'percent' and force_ylim_vals is not None:
                    lower_denom_fb = {str(k).strip().lower(): v for k, v in prot_denom_dict.items()}
                    raw_pct_fb = []
                    for raw_arr, o in zip(raw_dataset_v3, org_order_v3):
                        denom = lower_denom_fb.get(str(o).strip().lower(), 0)
                        raw_pct_fb.append((raw_arr / float(denom) * 100.0) if denom else np.zeros_like(raw_arr))
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, per_query_prot_counts,
                                 ylabel='Percentage of compartment total (per ciliary gene)',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES,
                                 dataset_override=raw_pct_fb, force_ylim=force_ylim_vals)
                else:
                    plot_grouped_violins(fig_v3, ax_v3, prot_values_dict, queries_prot, per_query_prot_counts,
                                 ylabel='Percentage of compartment total (per ciliary gene)',
                                 title=f'Ciliary Homologues: Proteins Normalized by {denom_source.title()} Compartment Total (per gene)',
                                 outpath=out_v3,
                                 labels_dict=None,
                                 normalize_by=prot_denom_dict,
                                 percent_scale=True,
                                 violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES)
            plt.close(fig_v3)

        # ── Enrichment: proteins ──────────────────────────────────────────────
        sp_denom_arr = np.array([prot_denom_dict.get(o, 0) for o in organelle_col], dtype=int)
        _enrichment_plots(
            k_vals     = human_counts.astype(int),
            K_vals     = sp_denom_arr,
            organelles = organelle_col.tolist(),
            out_dir    = 'plots_human_proteins',
            label      = 'Proteins',
            N_total    = N_total_prot,
            n_total    = n_total_prot,
            denom_source = denom_source,
        )

    # ── Gene normalized plots ────────────────────────────────────────────────

    if N_total_gene is not None and gene_values_dict:
        sp_denom_g  = np.array([gene_denom_dict.get(o, 0) for o in organelle_col_g], dtype=float)
        sp_fracs_g  = np.where(sp_denom_g > 0, human_counts_g / sp_denom_g, 0.0)

        # ── Plot 3g: Normalized (all compartments) ────────────────────────────
        print(f"\nCreating gene normalized plot ({denom_source}, all compartments)...")
        _sp_dict_g  = dict(zip(organelle_col_g, sp_fracs_g))
        _raw_dict_g = dict(zip(organelle_col_g, human_counts_g))
        _den_dict_g = dict(zip(organelle_col_g, sp_denom_g))
        _lbls3g = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(_raw_dict_g[o])}/{int(_den_dict_g[o])})"
                   for o in _sp_dict_g}
        _bys3g, _bvs3g, _bcs3g, _tys3g, _tls3g, _tcs3g, _lvls3g, _spans3g, _subspans3g = \
            _build_grouped_layout(_sp_dict_g, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls3g)
        fig3g, ax3g = plt.subplots(figsize=(max(10, len(_bys3g) * 0.45), 14))
        ax3g.bar(_bys3g, _bvs3g, color=_bcs3g, width=0.75)
        setattr(ax3g, 'move_labels_out', True)
        _apply_grouped_style(ax3g, _tys3g, _tls3g, _tcs3g, _lvls3g, _spans3g, _subspans3g)
        ax3g.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax3g.set_ylabel(f'Fraction of {denom_source} compartment total (%)')
        ax3g.set_title(f'Ciliary Homologues: Gene-Level Normalized by {denom_source.title()} Compartment Total (Human)',
                       fontsize=16, fontweight='bold', loc='right', pad=20)
        ax3g.grid(axis='y', alpha=0.3)
        ax3g.set_ylim(0, 1)
        yticks = np.linspace(0, 1, 11)
        ax3g.set_yticks(yticks)
        ax3g.set_yticklabels([f'{int(x*100)}' for x in yticks])
        plt.tight_layout()
        plt.savefig(f'plots_human_genes/organelle_human_gene_fraction_{denom_source}.png',
                    dpi=300, bbox_inches='tight')
        print(f"Plot saved as plots_human_genes/organelle_human_gene_fraction_{denom_source}.png")
        plt.close(fig3g)

        # ── Plot 4g: Normalized filtered (fraction >= 0.05) ───────────────────
        print(f"\nCreating gene normalized plot filtered ({denom_source}, fraction >= 0.05)...")
        mask_05_g = sp_fracs_g >= 0.05
        print(f"Filtered to {mask_05_g.sum()} gene compartments with normalized fraction >= 0.05")
        if mask_05_g.sum() > 0:
            _orgs_fg  = organelle_col_g[mask_05_g]
            _fracs_fg = sp_fracs_g[mask_05_g]
            _raw_fg   = human_counts_g[mask_05_g]
            _den_fg   = sp_denom_g[mask_05_g]
            _sp_dict4g = dict(zip(_orgs_fg, _fracs_fg))
            _lbls4g = {o: f"{DISPLAY.get(o, o.capitalize())} ({int(r)}/{int(d)})"
                       for o, r, d in zip(_orgs_fg, _raw_fg, _den_fg)}
            _bys4g, _bvs4g, _bcs4g, _tys4g, _tls4g, _tcs4g, _lvls4g, _spans4g, _subspans4g = \
                _build_grouped_layout(_sp_dict4g, GROUP_DEFS, GROUP_COLORS, labels_dict=_lbls4g)
            fig4g, ax4g = plt.subplots(figsize=(max(10, len(_bys4g) * 0.45), 14))
            ax4g.bar(_bys4g, _bvs4g, color=_bcs4g, width=0.75)
            setattr(ax4g, 'move_labels_out', True)
            _apply_grouped_style(ax4g, _tys4g, _tls4g, _tcs4g, _lvls4g, _spans4g, _subspans4g)
            ax4g.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
            ax4g.set_ylabel(f'Fraction of {denom_source} compartment total (%)')
            ax4g.set_title(f'Ciliary Homologues: Gene-Level Normalized >= 5% of {denom_source.title()} Compartment Total (Human)',
                          fontsize=16, fontweight='bold', loc='right', pad=20)
            ax4g.grid(axis='y', alpha=0.3)
            ax4g.set_ylim(0, 1)
            yticks = np.linspace(0, 1, 11)
            ax4g.set_yticks(yticks)
            ax4g.set_yticklabels([f'{int(x*100)}' for x in yticks])
            plt.tight_layout()
            plt.savefig(f'plots_human_genes/organelle_human_gene_fraction_filtered_{denom_source}.png',
                        dpi=300, bbox_inches='tight')
            print(f"Plot saved as plots_human_genes/organelle_human_gene_fraction_filtered_{denom_source}.png")
            plt.close(fig4g)
        else:
            print("No gene compartments passed the >= 0.05 threshold")

        # ── Violin 3g: Normalized gene fraction (all compartments) ───────────
        if per_query_gene_counts is not None:
            out_v3g = f'plots_human_genes/organelle_human_gene_fraction_violin_{denom_source}.png'
            fig_v3g, ax_v3g = plt.subplots(figsize=_presentation_figsize(len(gene_values_dict)))
            setattr(ax_v3g, 'move_labels_out', True)
            if per_query_gene_counts_fraction is not None:
                aligned_gene_frac = {q: per_query_gene_counts_fraction.get(q, {}) for q in queries_gene}
                plot_grouped_violins(fig_v3g, ax_v3g, gene_values_dict, queries_gene, aligned_gene_frac,
                             ylabel='Percentage of compartment total (per ciliary gene)',
                             title=f'Ciliary Homologues: Gene-Level Normalized by {denom_source.title()} Compartment Total (per gene)',
                             outpath=out_v3g,
                             percent_scale=True,
                             violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES)
            else:
                # fallback: use protein denominators if available
                denom_to_use = prot_denom_dict if prot_denom_dict else gene_denom_dict
                plot_grouped_violins(fig_v3g, ax_v3g, gene_values_dict, queries_gene, per_query_gene_counts,
                             ylabel='Percentage of compartment total (per ciliary gene)',
                             title=f'Ciliary Homologues: Gene-Level Normalized by {denom_source.title()} Compartment Total (per gene)',
                             outpath=out_v3g,
                             normalize_by=denom_to_use,
                             percent_scale=True,
                             violin_width=VIOLIN_WIDTH, font_scale=FONT_SCALE, pad_inches=PAD_INCHES)
            plt.close(fig_v3g)

        # ── Enrichment: genes ─────────────────────────────────────────────────
        sp_denom_g_arr = np.array([gene_denom_dict.get(o, 0) for o in organelle_col_g], dtype=int)
        _enrichment_plots(
            k_vals     = human_counts_g.astype(int),
            K_vals     = sp_denom_g_arr,
            organelles = organelle_col_g.tolist(),
            out_dir    = 'plots_human_genes',
            label      = 'Genes',
            N_total    = N_total_gene,
            n_total    = n_total_gene,
            denom_source = denom_source,
        )

print("\nAll plots complete.")
