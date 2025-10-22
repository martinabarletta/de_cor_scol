# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:24:39 2025

@author: matil
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get this script's folder (codes/)
script_dir = Path(__file__).parent

def distribution_mentions(df) : 
    # Filter out empty values in 'Text' column
    df_fr_filtered = df[df['Source'] != '']

    # Group by 'Text' and count occurrences
    df_counts = df_fr_filtered.groupby('Source').size().reset_index(name='Mentions')
    df_counts['Source'] = df_counts['Source'].str[:-4]

    #mentions_sum = df_counts['Mentions'].sum()

    # Rename 'old_name' to 'new_name'
    df_counts.rename(columns={'Source': 'texte'}, inplace=True)

    return df_counts

# Go up one level to project/
project_dir = script_dir.parent
data_dir = project_dir / "sheets/francais"

fr = pd.read_csv(data_dir / "toutes_mentions_corpus_detail_interdistance_fr.csv")
fr = fr.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
fr = distribution_mentions(fr)

data_dir = project_dir / "sheets/italien"

it = pd.read_csv(data_dir / "toutes_mentions_corpus_detail_interdistance_ita.csv")
it = it.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
it = distribution_mentions(it)

###Min, max, etendue ?
# ##diviser par niveau cad CE1 ou CE2 dans "texte"
level="CE1"
fr_CE1 = fr[fr['texte'].str.contains(level, case=False, na=False)]
it_CE1 = it[it['texte'].str.contains(level, case=False, na=False)]

level="CE2"
fr_CE2 = fr[fr['texte'].str.contains(level, case=False, na=False)]
it_CE2 = it[it['texte'].str.contains(level, case=False, na=False)]

# moy_ment_CE1 = df_CE1['Mentions'].mean()
# moy_ment_CE2 = df_CE2['Mentions'].mean()
# moy_ment = fr['Mentions'].mean()

# min_CE1 = df_CE1['Mentions'].min()
# min_CE2 = df_CE2['Mentions'].min()
# min_ment = fr['Mentions'].min()

# max_CE1 = df_CE1['Mentions'].max()
# max_CE2 = df_CE2['Mentions'].max()
# max_ment = fr['Mentions'].max()


fr["langue"] = "français"
fr["niveau"] = fr["texte"].str.extract(r"(CE1|CE2)")

it["langue"] = "italien"
it["niveau"] = it["texte"].str.extract(r"(CE1|CE2)")

df_fr_filtered = fr[["Mentions", "langue", "niveau"]].dropna()
df_it_filtered = it[["Mentions", "langue", "niveau"]].dropna()
df_combined = pd.concat([df_fr_filtered, df_it_filtered])
df_combined["niveau"] = pd.Categorical(df_combined["niveau"], categories=["CE1", "CE2"])

# Palette
full_palette = sns.color_palette("pastel")
selected_colors = [full_palette[6], full_palette[4]]

# Plot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=df_combined, x="niveau", y="Mentions", hue="langue", palette=selected_colors)
# fig.patch.set_facecolor('white')  # Fond blanc

label=14

x_offsets = {'français': -0.2, 'italien': 0.2}
x_base = {'CE1': 0, 'CE2': 1}

# Calcolo mediana per ogni gruppo
grouped = df_combined.groupby(['niveau', 'langue'])['Mentions'].median().reset_index()

grouped_stats = (
    df_combined.groupby(['niveau', 'langue'])['Mentions']
    .agg(['min', 'max', 'median', 'mean'])
    .reset_index()
)

highest_label = 0

for _, row in grouped_stats.iterrows():
    niveau = row['niveau']
    langue = row['langue']
    val_min = row['min']
    val_max = row['max']
    median = row['median']
    mean = row['mean']

    x = x_base[niveau] + x_offsets[langue]

    # --- Min label ---
    ax.text(
        x, val_min + 2, f'{int(val_min)}',
        ha='center', va='top', fontsize=label, color='black',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

    # --- Max label ---
    ax.text(
        x, val_max + 10, f'{int(val_max)}',
        ha='center', va='bottom', fontsize=label, color='black',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.8')
    )

    # --- Median label ---
    # ax.text(
    #     x, median + 10, f'Med: {int(median)}',
    #     ha='center', va='bottom', fontsize=label, color='black',
    #     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    # )

    # --- Mean label ---
    ax.text(
        x, mean + 10, f'Moy : {mean:.1f}',
        ha='center', va='bottom', fontsize=label, color='black',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

    # Track highest y position used
    highest_label = max(highest_label, val_max)

# --- Adjust plot margins dynamically ---
current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0], max(current_ylim[1], highest_label + 20))

# Titles and legend
plt.title("Distribution des mentions par texte", fontsize=16, pad=30)
plt.ylabel("Nombre de mentions", fontsize=14)
plt.xlabel("Niveau", fontsize=14)
ax.tick_params(axis='both', labelsize=14)

leg = ax.legend(title="Langue", prop={'size': 12})
leg.get_title().set_fontsize(12)

plt.tight_layout()
plt.show()

"""
# Posizioni dei box stimati manualmente
# seaborn assegna automaticamente le posizioni dei gruppi come: 0, 1 per 'CE1' e 'CE2'
# all’interno di ciascun gruppo, i livelli hue (français, italien) sono distanziati di +/- offset

for _, row in grouped.iterrows():
    niveau = row['niveau']
    langue = row['langue']
    median = row['Mentions']

    x = x_base[niveau] + x_offsets[langue]
    ax.text(x, median + 10, f'{int(median)}',
            ha='center', va='bottom', fontsize=label, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

# Calcola min e max per ogni box
grouped_extremes = df_combined.groupby(['niveau', 'langue'])['Mentions'].agg(['min', 'max']).reset_index()

for _, row in grouped_extremes.iterrows():
    niveau = row['niveau']
    langue = row['langue']
    val_min = row['min']
    val_max = row['max']

    x = x_base[niveau] + x_offsets[langue]

    # Etichetta sotto il minimo
    ax.text(x, val_min - 10, f'{int(val_min)}',
            ha='center', va='top', fontsize=label, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

    # Etichetta sopra il massimo
    ax.text(x, val_max + 10, f'{int(val_max)}',
            ha='center', va='bottom', fontsize=label, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=1'))

# Titoli e legenda
plt.title("Comparaison de la longueur des textes \npar langue et par niveau", fontsize=16, pad=30)
plt.ylabel("Nombre de tokens par texte", fontsize=14)
plt.xlabel("Niveau", fontsize=14)
ax.tick_params(axis='both', labelsize=14)

leg = ax.legend(title="Langue", prop={'size': 12})
leg.get_title().set_fontsize(12)

plt.tight_layout()
plt.show()"""