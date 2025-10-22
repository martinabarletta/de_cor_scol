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

current_dir = Path(__file__).parent
#file_path = current_dir.parent / "sheets/francais/toutes_mentions_corpus_detail_interdistance_fr.csv"
file_path = current_dir.parent / "sheets/italien/toutes_mentions_corpus_detail_interdistance_ita.csv"

df = pd.read_csv(file_path)

###prendere il valore massimo di ogni combinazione Source, tag e fare lista Source, tag, valore massimo
###scrivere in un csv

# 1️⃣ Raggruppa per Source e tag, e prendi il massimo del valore
df_max = df.groupby(['Source', 'tag'], as_index=False)['tag_occurrences'].max()

# 2️⃣ (Facoltativo) Ordina per Source e tag
df_max = df_max.sort_values(['Source', 'tag'])

# 3️⃣ Scrivi in un file CSV
df_max.to_csv('./output_it.csv', index=False)

##per ottenere la lunghezza delle catene, filtrare via tag_occurrences 1 e 2 !!!
df_filtered = df_max[~df_max['tag_occurrences'].isin([1, 2])]

moyenne = df_filtered['tag_occurrences'].mean()
df_filtered["langue"] = "italien"
df_filtered["niveau"] = df_filtered["Source"].str.extract(r"(CE1|CE2)")

#Filtra solo le righe dove niveau = 'CE1'
df_ce1 = df_filtered[df_filtered['niveau'] == 'CE1']

#Calcola la media di tag_occurrences
media_CE1 = df_ce1['tag_occurrences'].mean()

#Filtra solo le righe dove niveau = 'CE1'
df_ce2 = df_filtered[df_filtered['niveau'] == 'CE2']

#Calcola la media di tag_occurrences
media_CE2 = df_ce2['tag_occurrences'].mean()

df_filtered.rename(columns={'tag_occurrences': 'len_chaines'}, inplace=True)

# Palette Seaborn "pastel"
colors = sns.color_palette("pastel", 20)  # 20 colori diversi per 20 bin
plt.figure(figsize=(12, 8))

# Crea l'istogramma
counts, bins, patches = plt.hist(df_filtered['len_chaines'], bins=20, edgecolor='black', alpha=0.8)

# Applica colori diversi secondo l'altezza (frequenza)
for count, patch, color in zip(counts, patches, colors):
    patch.set_facecolor(color)
    # Aggiunge etichetta sopra la barra
    plt.text(patch.get_x() + patch.get_width()/2, count + 0.05, int(count), 
             ha='center', va='bottom', fontsize=14)

plt.title('Distribuzione di len_chaines con palette pastel')
plt.xlabel('Longueur des chaines')
plt.ylabel('Frequence')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#######################

# Crea figure e due subplot affiancati
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, df, title in zip(axes, [df_ce1, df_ce2], ['CE1', 'CE2']):
    counts, bins, patches = ax.hist(df['tag_occurrences'], bins=20, edgecolor='black', alpha=0.8)
    
    # Applica colori e aggiungi etichette
    for count, patch, color in zip(counts, patches, colors):
        patch.set_facecolor(color)
        ax.text(patch.get_x() + patch.get_width()/2, count + 0.05, int(count),
                ha='center', va='bottom', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel('Longueur des chaines')
    ax.set_ylabel('Frequence')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle('Distribution de la longueur des chaines - corpus italien', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Définir les bins
bins = np.arange(0, max(df_ce1['tag_occurrences'].max(), df_ce2['tag_occurrences'].max()) + 2) - 0.5

# Comptage pour chaque bin
counts_ce1, _ = np.histogram(df_ce1['tag_occurrences'], bins=bins)
counts_ce2, _ = np.histogram(df_ce2['tag_occurrences'], bins=bins)

# Positions des barres
x = np.arange(len(counts_ce1))
width = 0.45  # largeur des barres augmentée

fig, ax = plt.subplots(figsize=(12, 6))

# Barres côte à côte
bar1 = ax.bar(x - width/2, counts_ce1, width, label='CE1', color=colors[4], edgecolor='black')
bar2 = ax.bar(x + width/2, counts_ce2, width, label='CE2', color=colors[2], edgecolor='black')

# Ajouter les valeurs au-dessus des barres, sauf si la valeur est 0
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # seulement si la hauteur est >0
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, int(height),
                    ha='center', va='bottom', fontsize=9)

# Labels et titre
ax.set_xticks(x)
ax.set_xticklabels([str(int(b+0.5)) for b in bins[:-1]])  # afficher les longueurs exactes
ax.set_xlabel('Longueur des chaines')
ax.set_ylabel('Frequence')
ax.set_title('Distribution de la longueur des chaines - corpus italien')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
plt.show()


#=== con linee 
# Définir les bins
bins = np.arange(0, max(df_ce1['tag_occurrences'].max(), df_ce2['tag_occurrences'].max()) + 2) - 0.5

# Comptage pour chaque bin
counts_ce1, _ = np.histogram(df_ce1['tag_occurrences'], bins=bins)
counts_ce2, _ = np.histogram(df_ce2['tag_occurrences'], bins=bins)

# Positions des barres
x = np.arange(len(counts_ce1))
width = 0.45  # largeur des barres augmentée

fig, ax = plt.subplots(figsize=(12, 6))

# Barres côte à côte
bar1 = ax.bar(x - width/2, counts_ce1, width, label='CE1', color=colors[4], edgecolor='black')
bar2 = ax.bar(x + width/2, counts_ce2, width, label='CE2', color=colors[2], edgecolor='black')

# Ajouter les valeurs au-dessus des barres
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, int(height),
                    ha='center', va='bottom', fontsize=9)

# ==== AJOUTER LES COURBES LINEAIRES ====
# Courbe CE1
ax.plot(x - width/2, counts_ce1, color=colors[4], marker='o', linestyle='-', linewidth=2, label='Courbe CE1')

# Courbe CE2
ax.plot(x + width/2, counts_ce2, color=colors[2], marker='o', linestyle='--', linewidth=2, label='Courbe CE2')

# Labels et titre
ax.set_xticks(x)
ax.set_xticklabels([str(int(b+0.5)) for b in bins[:-1]])
ax.set_xlabel('Longueur des chaines')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution de la longueur des chaines - corpus italien')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


