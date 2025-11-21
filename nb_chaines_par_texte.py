# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:21:00 2025

@author: matil

Calcule le nombre de chaines par texte
prends en entrée le fichier len_stats.csv (obtenu avec le code XXX) 
qui contient un texte par ligne et une chaine par colonne (chaine plurielle incluses)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
# -------------------------------------------------------------------
# PROCESSING FUNCTIONS
# -------------------------------------------------------------------

def load_and_compute_max(file_path):
    """Charge le CSV, groupe par Source/tag, garde le max de tag_occurrences."""
    df = pd.read_csv(file_path)
    df_max = df.groupby(['Source', 'tag'], as_index=False)['tag_occurrences'].max()
    df_max = df_max.sort_values(['Source', 'tag'])
    df_max.rename(columns={'tag_occurrences': 'len_chaines'}, inplace=True)
    return df_max


def filter_chains(df):
    """Élimine les chaines de longueur 1 et 2."""
    return df[~df['len_chaines'].isin([1, 2])].copy()


def add_metadata(df, langue):
    """Ajoute langue et niveau extrait du champ Source."""
    df["langue"] = langue
    df["niveau"] = df["Source"].str.extract(r"(CE1|CE2)")
    return df


def split_by_level(df):
    """Sépare en deux dataframes CE1 et CE2."""
    return df[df["niveau"] == "CE1"], df[df["niveau"] == "CE2"]


# -------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------

def plot_histogram(df, langue):
    """Histogramme global (version pastel)."""
    colors = sns.color_palette("pastel", 20)

    plt.figure(figsize=(12, 8))
    counts, bins, patches = plt.hist(df['len_chaines'], bins=20, edgecolor='black', alpha=0.8)

    for count, patch, color in zip(counts, patches, colors):
        patch.set_facecolor(color)
        if count > 0:
            plt.text(
                patch.get_x() + patch.get_width()/2,
                count + 0.05,
                int(count),
                ha='center', fontsize=12
            )

    plt.title(f"Distribution de la longueur des chaines – corpus {langue}", fontsize=16)
    plt.xlabel("Longueur des chaines", fontsize=14)
    plt.ylabel("Fréquence", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_two_histograms(df_ce1, df_ce2, langue):
    """Histogrammes CE1 et CE2 côte à côte."""
    colors = sns.color_palette("pastel", 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, df, title in zip(axes, [df_ce1, df_ce2], ["CE1", "CE2"]):
        counts, bins, patches = ax.hist(df['len_chaines'], bins=20, edgecolor='black', alpha=0.8)
        for count, patch, color in zip(counts, patches, colors):
            patch.set_facecolor(color)
            if count > 0:
                ax.text(
                    patch.get_x() + patch.get_width()/2,
                    count + 0.05,
                    int(count),
                    ha='center', fontsize=10
                )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longueur des chaines", fontsize=12)
        ax.set_ylabel("Fréquence", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Distribution CE1 vs CE2 – corpus {langue}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_bars_with_lines(df_ce1, df_ce2, langue):
    """Barplot + curves CE1/CE2."""
    colors = sns.color_palette("pastel", 20)

    # Prepare bins
    bins = np.arange(
        0, max(df_ce1['len_chaines'].max(), df_ce2['len_chaines'].max()) + 2
    ) - 0.5

    counts_ce1, _ = np.histogram(df_ce1['len_chaines'], bins=bins)
    counts_ce2, _ = np.histogram(df_ce2['len_chaines'], bins=bins)

    x = np.arange(len(counts_ce1))
    width = 0.45

    fig, ax = plt.subplots(figsize=(12, 6))

    bar1 = ax.bar(x - width/2, counts_ce1, width, label="CE1", color=colors[4], edgecolor="black")
    bar2 = ax.bar(x + width/2, counts_ce2, width, label="CE2", color=colors[2], edgecolor="black")

    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, int(height),
                        ha="center", fontsize=9)

    # Add lines
    # ax.plot(x - width/2, counts_ce1, marker="o", color=colors[4], linewidth=2)
    # ax.plot(x + width/2, counts_ce2, marker="o", linestyle="--", color=colors[2], linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b + 0.5)) for b in bins[:-1]], fontsize=11)

    ax.set_xlabel("Longueur des chaines", fontsize=13)
    ax.set_ylabel("Fréquence", fontsize=13)
    ax.set_title(f"Distribution de la longueur des chaînes – corpus {langue}", fontsize=16)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# MASTER PIPELINE
# -------------------------------------------------------------------

def process_file(file_path, langue, save_csv=True):
    """Pipeline complet pour un fichier donné."""
    df_max = load_and_compute_max(file_path)
    df_clean = filter_chains(df_max)
    df_clean = add_metadata(df_clean, langue)

    df_ce1, df_ce2 = split_by_level(df_clean)

    if save_csv:
        df_clean.to_csv(f"output_{langue}.csv", index=False)

    # ↓ CHOOSE WHICH PLOTS YOU WANT ↓
    plot_histogram(df_clean, langue)
    plot_two_histograms(df_ce1, df_ce2, langue)
    plot_bars_with_lines(df_ce1, df_ce2, langue)

    return df_clean, df_ce1, df_ce2


# -------------------------------------------------------------------
# EXECUTION EXAMPLE
# -------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = Path(__file__).parent

    file_italien = script_dir.parent / "sheets/italien/toutes_mentions_corpus_detail_interdistance_ita.csv"
    file_francais = script_dir.parent / "sheets/francais/toutes_mentions_corpus_detail_interdistance_fr.csv"

    process_file(file_italien, langue="italien")
    process_file(file_francais, langue="français")



def compute_chain_lengths(file_path):
    """find max chain per text"""
    df = pd.read_csv(file_path)
    df['Source'] = df['Source'].str[:-4]
    # Max per Source/tag
    df_max = df.groupby(['Source', 'tag'], as_index=False)['tag_occurrences'].max()

    # Remove lengths 1–2
    df_max = df_max[~df_max['tag_occurrences'].isin([1, 2])]

    # Rename for clarity
    df_max = df_max.rename(columns={'tag_occurrences': 'len_chaines'})
    return df_max

def enrich_metadata(df, langue):
    df['langue'] = langue
    df['niveau'] = df['Source'].str.extract(r"(CE1|CE2)")
    return df

def merge_text_lengths(df_chains, file_lengths):
    df_len = pd.read_csv(file_lengths)
    df_len = df_len[['texte', 'nbTokNoPunct']]
    df_chains = df_chains.rename(columns={'Source': 'texte'})

    # Merge on Source
    df_merged = df_chains.merge(df_len, on='texte', how='left')

    # Check missing lengths
    if df_merged['nbTokNoPunct'].isna().any():
        missing = df_merged[df_merged['nbTokNoPunct'].isna()]['texte'].unique()
        print("⚠️ Missing text lengths for:", missing)

    return df_merged

def normalize_chain_lengths(df):
    df['len_normalised'] = df['len_chaines'] / df['nbTokNoPunct']
    return df

def split_levels(df):
    df_ce1 = df[df['niveau'] == 'CE1']
    df_ce2 = df[df['niveau'] == 'CE2']
    return df_ce1, df_ce2

def plot_histograms(df_ce1, df_ce2, langue, variable="len_normalised"):
    colors = sns.color_palette("pastel", 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, df, level in zip(axes, [df_ce1, df_ce2], ['CE1', 'CE2']):
        counts, bins, patches = ax.hist(df[variable], bins=20,
                                        edgecolor='black', alpha=0.8)

        for count, patch, color in zip(counts, patches, colors):
            patch.set_facecolor(color)
            if count > 0:
                ax.text(patch.get_x() + patch.get_width()/2,
                        count + 0.05,
                        int(count),
                        ha='center', fontsize=8)

        ax.set_title(f"{langue} – {level}")
        ax.set_xlabel("Longueur normalisée des chaînes")
        ax.set_ylabel("Fréquence")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Distribution des longueurs normalisées des chaînes – {langue}",
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def process_language_chains(
    file_chains,
    file_lengths,
    langue="français"
):
    # Step 1 – compute max lengths
    df = compute_chain_lengths(file_chains)

    # Step 2 – enrich (CE1/CE2 + langue)
    df = enrich_metadata(df, langue)

    # Step 3 – merge with text lengths
    df = merge_text_lengths(df, file_lengths)

    # Step 4 – normalisation
    df = normalize_chain_lengths(df)

    # Step 5 – split CE1 / CE2
    df_ce1, df_ce2 = split_levels(df)

    # Step 6 – plots
    plot_histograms(df_ce1, df_ce2, langue)

    return df

if __name__ == "__main__":
    base = Path(__file__).parent.parent

    df_it = process_language_chains(
        file_chains = base/"sheets/italien/toutes_mentions_corpus_detail_interdistance_ita.csv",
        file_lengths = base/"sheets/italien/corpus_italien.csv",
        langue="italien"
    )

    df_fr = process_language_chains(
        file_chains = base/"sheets/francais/toutes_mentions_corpus_detail_interdistance_fr.csv",
        file_lengths = base/"sheets/francais/corpus_francais_v3.csv",
        langue="français"
    )

#subplot fr + ita
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.histplot(df_fr['len_normalised'], bins=20, color='skyblue', ax=axes[0])
axes[0].set_title("Français")

sns.histplot(df_it['len_normalised'], bins=20, color='lightcoral', ax=axes[1])
axes[1].set_title("Italien")

for ax in axes:
    ax.set_xlabel("Longueur normalisée")
    ax.set_ylabel("Fréquence")
plt.tight_layout()
plt.show()



# Define common bins
bins = np.linspace(0, max(df_fr['len_normalised'].max(), df_it['len_normalised'].max()), 20)
counts_fr, _ = np.histogram(df_fr['len_normalised'], bins=bins)
counts_it, _ = np.histogram(df_it['len_normalised'], bins=bins)

x = np.arange(len(counts_fr))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(x - width/2, counts_fr, width, label='Français', color='skyblue', edgecolor='black')
bar2 = ax.bar(x + width/2, counts_it, width, label='Italien', color='lightcoral', edgecolor='black')

# Add counts above bars
for bars in [bar1, bar2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, int(h),
                    ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"{b:.2f}" for b in bins[:-1]], rotation=45)
ax.set_xlabel("Longueur normalisée")
ax.set_ylabel("Fréquence")
ax.set_title("Comparaison des longueurs max per texte normalisées entre français et italien")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


##avec KDE curves###############################################################################
# Calcola il centro e la larghezza dei bin

# Bin e larghezza
bins = np.linspace(0, max(df_fr['len_normalised'].max(), df_it['len_normalised'].max()), 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
width = bin_width * 0.4

# Istogrammi
counts_fr, _ = np.histogram(df_fr['len_normalised'], bins=bins)
counts_it, _ = np.histogram(df_it['len_normalised'], bins=bins)

fig, ax = plt.subplots(figsize=(12,6))

# Barre
bar1 = ax.bar(bin_centers - width/2, counts_fr, width, label='Français', color='skyblue', edgecolor='black')
bar2 = ax.bar(bin_centers + width/2, counts_it, width, label='Italien', color='lightcoral', edgecolor='black')

# Numeri sopra le barre
for bars in [bar1, bar2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, int(h),
                    ha='center', va='bottom', fontsize=9)

# KDE scalata
def scaled_kde(data, counts, bins):
    kde = gaussian_kde(data, bw_method=0.5)
    x = np.linspace(bins[0], bins[-1], 500)
    y = kde(x)
    # Scala la KDE come l'istogramma
    y_scaled = y * len(data) * (bins[1] - bins[0])
    return x, y_scaled

x_fr, y_fr = scaled_kde(df_fr['len_normalised'], counts_fr, bins)
x_it, y_it = scaled_kde(df_it['len_normalised'], counts_it, bins)

ax.plot(x_fr, y_fr, color='blue', linewidth=2, label='KDE Français')
ax.plot(x_it, y_it, color='red', linewidth=2, linestyle='--', label='KDE Italien')

# X-ticks
ax.set_xticks(bin_centers)
ax.set_xticklabels([f"{b:.2f}" for b in bin_centers], rotation=45)

ax.set_xlabel("Longueur normalisée", fontsize=12)
ax.set_ylabel("Fréquence", fontsize=12)
ax.set_title("Comparaison des longueurs max per texte normalisées entre français et italien", fontsize=14)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
