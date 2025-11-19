# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 20:05:38 2025

@author: matil

plots pour partie chapitre 3 sur l'interdistance entre mentions
NEW la colonne "distance" indique la distance entre maillons du texte
la colonne interdistance indique la distance entre maillons de la même chaine

"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent

###français
file = project_dir/'sheets/francais/toutes_mentions_corpus_detail_interdistance_fr.csv'

###italien
#file = project_dir/'annotations_italien/toutes_mentions_corpus_detail_interdistance_ita.csv'
df = pd.read_csv(file)

#####interdistance = entre mentions du meme referent
indice = "interdistance" #ou "interdistance

sum_df = df.groupby(['Source', 'tag'])[indice].sum().reset_index()
max_df = df.groupby(['Source', 'tag'])['tag_occurrences'].max().reset_index()
res = pd.merge(sum_df, max_df, on=['Source', 'tag'])
result = res[~res['tag_occurrences'].isin([1, 2])]

###ratio interdistance inspiré de Rousier-Vercruyssen et Landragin 2019
result['Interdistance_ratio'] = result["interdistance"] / result['tag_occurrences']

result.to_csv(project_dir/'annotations_francais/interdistance_fr.csv', index=False)

########################""INTERDISTANCE PUR###################################
#calcul min, max, moyenne tot puis par niveau
# 1. Min and max for each numeric column
min_max_all = result.describe().loc[['min', 'max']]
print("Min & Max for each column:")
print(min_max_all)

filtered_stats = result['interdistance'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for French INTERDISTANCE PUR :")
print(filtered_stats)
#???
filtered_stats["CV"] = filtered_stats["std"] / filtered_stats["mean"]


# 2. Filter rows where Source contains a string
mask = result['Source'].str.contains('CE1', case=False, na=False)
filtered_statsCE1 = result.loc[mask, 'interdistance'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for CE1:")
print(filtered_statsCE1)

# 2. Filter rows where Source contains a string
mask = result['Source'].str.contains('CE2', case=False, na=False)
filtered_statsCE2 = result.loc[mask, 'interdistance'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for CE2:")
print(filtered_statsCE2)

#######INTERDISTANCE RATIO#####################################################
#calcul min, max, moyenne tot puis par niveau
# 1. Min and max for each numeric column
min_max_all = result.describe().loc[['min', 'max']]
print("Min & Max for each column:")
print(min_max_all)

filtered_stats = result['Interdistance_ratio'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for French :")
print(filtered_stats)
#???
filtered_stats["CV"] = filtered_stats["std"] / filtered_stats["mean"]


# 2. Filter rows where Source contains a string
mask = result['Source'].str.contains('CE1', case=False, na=False)
filtered_statsCE1 = result.loc[mask, 'Interdistance_ratio'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for CE1:")
print(filtered_statsCE1)

# 2. Filter rows where Source contains a string
mask = result['Source'].str.contains('CE2', case=False, na=False)
filtered_statsCE2 = result.loc[mask, 'Interdistance_ratio'].agg(['min', 'max', 'mean', 'std'])
print("\nStats for CE2:")
print(filtered_statsCE2)

####PLOT SUR RATIO INTERDISTANCE###############################################
# Parameters
bar_width = 1.2  # proportion of bin taken by bar (0 < bar_width ≤ 1)
gap_factor = 1.8  # space between bars (bigger = more spacing)
bin_width = 1
max_val = result["Interdistance_ratio"].max()

# Build bins
bin_edges = []
start = 0
while start < max_val:
    bin_edges.append(start)
    start += bin_width
bin_edges.append(start)

# Histogram
counts, edges = np.histogram(result['Interdistance_ratio'], bins=bin_edges)

# Centers for bars (apply gap factor by shifting them apart)
centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)])
centers = centers * gap_factor  # stretch spacing between bars

# Get pastel palette colors
palette = sns.color_palette("pastel", 6)  # 4 colors for 4 ranges
c1, c2, c3, c4, c5, c6 = palette  # unpack

# Assign colors based on height ranges
colors = []
for count in counts:
    if 1 <= count <= 10:
        colors.append(c1)
    elif 11 <= count <= 20:
        colors.append(c2)
    elif 21 <= count <= 30:
        colors.append(c3)
    elif 31 <= count <= 40:
        colors.append(c4)  # default for other ranges
    elif 41 <= count <= 50:
        colors.append(c5)
    else:
        colors.append(c6)


plt.figure(figsize=(14, 6))

# Plot bars
plt.bar(
    x=centers,
    height=counts,
    width=bar_width,  # control thickness independently
    align='center',
    edgecolor='black',
    color=colors
)

# Add counts above bars (only if > 0)
for x, y in zip(centers, counts):
    if y > 0:
        plt.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=12)

# X-axis ticks every 5 units (scaled to gap factor)
tick_positions = np.arange(0, max_val + 5, 5) * gap_factor
plt.xticks(ticks=tick_positions, labels=np.arange(0, max_val + 5, 5), fontsize=12)
plt.grid(which='major', zorder=-1.0)
plt.xlabel("Taux de ratio interdistance")
plt.ylabel("Nombre des chaines")
plt.title("")
plt.show()

##################################################################################
#comparaison CE1 vs CE2
# --- PARAMETERS ---
bar_width = 1.2
gap_factor = 1.8
bin_width = 1

# --- DEFINE COLORS (pastel palette) ---
palette = sns.color_palette("pastel", 6)
c1, c2, c3, c4, c5, c6 = palette

def plot_interdistance_hist(ax, data, title=""):
    """Draw histogram for given subset on provided axes"""
    max_val = data["Interdistance_ratio"].max()

    # Build bins
    bin_edges = np.arange(0, max_val + bin_width, bin_width)
    counts, edges = np.histogram(data["Interdistance_ratio"], bins=bin_edges)

    # Compute centers and stretch spacing
    centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)])
    centers = centers * gap_factor

    # Assign colors based on count ranges
    colors = []
    for count in counts:
        if 1 <= count <= 10:
            colors.append(c1)
        elif 11 <= count <= 20:
            colors.append(c2)
        elif 21 <= count <= 30:
            colors.append(c3)
        elif 31 <= count <= 40:
            colors.append(c4)
        elif 41 <= count <= 50:
            colors.append(c5)
        else:
            colors.append(c6)

    # Plot
    ax.bar(
        x=centers,
        height=counts,
        width=bar_width,
        align='center',
        edgecolor='black',
        color=colors
    )

    # Add count labels
    for x, y in zip(centers, counts):
        if y > 0:
            ax.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=11)

    # Adjust ticks
    tick_positions = np.arange(0, max_val + 5, 5) * gap_factor
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(np.arange(0, max_val + 5, 5), fontsize=12)
    ax.grid(which='major', zorder=-1.0)
    ax.set_xlabel("Taux de ratio interdistance", fontsize=12)
    ax.set_ylabel("Nombre des chaines", fontsize=12)
    ax.set_title(title, fontsize=14)

# --- SPLIT DATA ---
#result['Source'].str.contains('CE1', case=False, na=False)
df_ce1 = result[result["Source"].str.contains('CE1', case=False, na=False)]
df_ce2 = result[result["Source"].str.contains('CE2', case=False, na=False)]

# --- CREATE SUBPLOTS ---
fig, axes = plt.subplots(2,1, figsize=(14, 12), sharey=True)

plot_interdistance_hist(axes[0], df_ce1, title="CE1")
plot_interdistance_hist(axes[1], df_ce2, title="CE2")

plt.tight_layout()
plt.show()
##################################################################################
#représenter la variation dans l'interdistance ???