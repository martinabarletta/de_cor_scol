# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:22:49 2025

@author: matil
"""
# -*- coding: utf-8 -*-
"""
Script per contare le anafore per tipo e per livello scolastico
Input: CSV con colonne come 'Source', 'tag', 'origin' o simili
Output: DataFrame con conteggi e plot a barre verticali e orizzontali
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config
# -------------------------
sheet_groups = {
    "sn_def": "SN défini",
    "sn_indef": "SN indéfini",
    "verb": "Anaphore Zéro",
    "pron": "Pronom",
    "propn": "Nom Propre",
    "det_poss": "SN possessif",
    "sn_poss": "SN possessif",
    "sn_dem": "SN démonstratif",
    "sn_no_det": "SN sans déterminant",
    "numerals": "Autre",
    "autre": "Autre"
}

categories_order = ["Pronom", "SN défini", "SN indéfini",
                    "SN possessif", "SN démonstratif", "SN sans déterminant",
                    "Nom Propre", "Anaphore Zéro", "Autre"]

# Paths

# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent
csv_path = project_dir / "sheets/francais/anaphores_fr.csv"  # sostituire con il CSV corretto
plots_dir = project_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def add_school_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["School_Level"] = df["Source"].astype(str).apply(
        lambda s: "CE1" if "CE1" in s else ("CE2" if "CE2" in s else pd.NA)
    )
    return df

def map_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = df["origin"].map(sheet_groups).fillna("Autre")
    return df

# -------------------------
# Conteggi
# -------------------------
def count_anaphoras(df: pd.DataFrame) -> pd.DataFrame:
    df = add_school_level(df)
    df = map_category(df)
    df = df[df['School_Level'].notna()]  # escludo righe senza livello
    grouped = df.groupby(['Category', 'School_Level']).size().reset_index(name='Count')
    return grouped

# -------------------------
# Plot
# -------------------------
# def plot_counts_bar(df_counts: pd.DataFrame, horizontal=False, title="Conteggio anafore"):
#     df_pivot = df_counts.pivot(index='Category', columns='School_Level', values='Count').fillna(0)
#     df_pivot = df_pivot[categories_order] if horizontal else df_pivot
#     df_pivot = df_pivot.sort_index()
    
#     if horizontal:
#         ax = df_pivot.plot(kind='barh', stacked=True, figsize=(12,8), edgecolor='black', color=sns.color_palette("pastel")[0:2])
#     else:
#         ax = df_pivot.plot(kind='bar', stacked=True, figsize=(12,8), edgecolor='black', color=sns.color_palette("pastel")[0:2])
    
#     plt.title(title, fontsize=16)
#     plt.ylabel('Count' if not horizontal else '')
#     plt.xlabel('Category' if not horizontal else 'Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(plots_dir / f"{title.replace(' ', '_')}.png", dpi=1200)
#     plt.show()

# -------------------------
# Main
# -------------------------

df = pd.read_csv(csv_path)
counts_df = count_anaphoras(df)
print("Conteggi per categoria e livello:")
print(counts_df)

# plot_counts_bar(counts_df, horizontal=False, title="Anaphoras per Category and Level")
# plot_counts_bar(counts_df, horizontal=True, title="Anaphoras per Category and Level Horizontal")



