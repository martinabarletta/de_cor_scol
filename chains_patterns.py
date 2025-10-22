# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:06:29 2025

@author: matil

draw text patterns 
plot tokenBegin instead of interdistance 
start at token 1 (not 0)


permet de créer un scatterplot par texte qui décrit une chaine par ligne et 
qui contient la succession des mentions par type (un symbole différent par type de mention)
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np

def load_and_process_excel(file_path):
    # Load all sheets
    sheets = pd.read_excel(file_path, sheet_name=None)

    # Add SheetName column and concatenate
    for sheet_name, df in sheets.items():
        df['SheetName'] = sheet_name
        sheets[sheet_name] = df

    combined_df = pd.concat(sheets.values(), ignore_index=True)

    # Ensure Source and tag exist and sort
    if 'Source' not in combined_df.columns or 'tag' not in combined_df.columns:
        raise ValueError("Expected 'Source' and 'tag' columns in input Excel sheets.")
    
    combined_df.sort_values(by=['Source', 'tag', 'tag_occurrences'], inplace=True)

    return combined_df


# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent


# Load the Excel file
file_path_fr = project_dir/"annotations_francais/typologie_mentions_fr_V4.xlsx"
file_path_it = project_dir/"annotations_italien/ita_types_v2.xlsx"


# === Run the complete process ===
file_path = file_path_fr  #
output_folder = project_dir/"type_plots/"
# Load, process, compute, and visualize
df = load_and_process_excel(file_path)

# === === === === === === === === === === === === === === === === === === ===
# Clean tokenBegin column
df['tokenBegin'] = pd.to_numeric(df['tokenBegin'], errors='coerce')
df = df.dropna(subset=['tokenBegin'])

# Ensure integer token positions
df['tokenBegin'] = df['tokenBegin'].astype(int)

# Unique mention types and markers/colors
mention_types = df['SheetName'].unique()
markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']  # 
colors = plt.cm.tab10.colors  # categorical color palette

type_to_style = {
    mtype: (markers[i % len(markers)], colors[i % len(colors)])
    for i, mtype in enumerate(mention_types)
}

# inside your existing loop over sources, replace the plotting block with this

# Iterate over each Source
for source in df['Source'].unique():
    subset_source = df[df['Source'] == source]

    # ORDER tags by their first mention (minimum tokenBegin) - computed per source
    tag_order = (
        subset_source.groupby("tag")["tokenBegin"]
        .min()
        .sort_values()         # earliest first
        .index.tolist()
    )

    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # Plot using the sorted tag_order
    for tag_idx, tag in enumerate(tag_order):
        subset_tag = subset_source[subset_source['tag'] == tag]
        for _, row in subset_tag.iterrows():
            marker, color = type_to_style[row['SheetName']]
            ax.scatter(row['tokenBegin'], tag_idx, marker=marker, color=color, s=100)

    # Formatting
    ax.set_yticks(range(len(tag_order)))
    ax.set_yticklabels(tag_order)
    ax.set_xlabel("Token position in text")
    ax.set_ylabel("Referent chains (tags)")
    ax.set_title(f"Mention Type Distribution for Source: {source}")

    # Put the earliest (index 0) on TOP
    ax.invert_yaxis()

    # Legend (same as yours)
    handles = [plt.Line2D([], [], color=color, marker=marker, linestyle='-', markersize=10)
               for mtype, (marker, color) in type_to_style.items()]
    labels = list(type_to_style.keys())
    ax.legend(handles, labels, title="Mention Types", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Ensure output folder exists and use a safe filename
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{source[:-4]}.jpg")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()   # IMPORTANT: close the figure so subsequent iterations start fresh
