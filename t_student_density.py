# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:49:58 2025

@author: matil
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path

script_dir = Path(__file__).parent
project_dir = script_dir.parent

res_fr = project_dir/"annotations_francais/density_fr.csv"
res_it = project_dir/"annotations_italien/density_ita_v2.csv"

# Carica i file CSV
df1 = pd.read_csv(res_fr)   # DF1 = francese
df2 = pd.read_csv(res_it)   # DF2 = italiano


# === Funzione annotazione boxplot ===
def annotate_boxplot(ax, data, bp):
    """Aggiunge etichette numeriche (Q1, Q3, median, whiskers, outliers) ai boxplot"""
    for i, y in enumerate(data, start=1):
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        iqr = q3 - q1
        whis_low = np.min(y[y >= q1 - 1.5 * iqr])
        whis_high = np.max(y[y <= q3 + 1.5 * iqr])
        outliers = y[(y < whis_low) | (y > whis_high)]

        # Etichette
        ax.text(i+0.1, med, f'{med:.2f}', va='center', color='red')
        ax.text(i+0.1, q1, f'{q1:.2f}', va='center', color='blue')
        ax.text(i+0.1, q3, f'{q3:.2f}', va='center', color='blue')
        ax.text(i+0.1, whis_low, f'{whis_low:.2f}', va='center', color='green')
        ax.text(i+0.1, whis_high, f'{whis_high:.2f}', va='center', color='green')
        for f in outliers:
            ax.text(i+0.1, f, f'{f:.2f}', va='center', color='purple')

############################################################################################
# Francese vs Italiano
gruppo1 = df1['density'] # fr
gruppo2 = df2['density'] # ita
statistica, p_value = ttest_ind(gruppo1, gruppo2, equal_var=False)
print(f'Statistica t: {statistica}')
print(f'P-value: {p_value}')

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([gruppo1, gruppo2], labels=['français', 'italien'], patch_artist=True)
annotate_boxplot(ax, [gruppo1.values, gruppo2.values], bp)

ax.set_title(f'Comparaison densité référentielle entre français et italien\nTest t: p = {p_value:.3g}')
ax.set_ylabel('Densité référentielle')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

############################################################################################
# CE1 vs CE2 - Français
gruppo1 = df1[df1['group'] == 'CE1']['density']
gruppo2 = df1[df1['group'] == 'CE2']['density']
statistica, p_value = ttest_ind(gruppo1, gruppo2, equal_var=False)
print(f'Statistica t: {statistica}')
print(f'P-value: {p_value}')

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([gruppo1, gruppo2], labels=['CE1', 'CE2'], patch_artist=True)
annotate_boxplot(ax, [gruppo1.values, gruppo2.values], bp)

ax.set_title(f'Comparaison densité référentielle entre CE1 et CE2 français\nTest t: p = {p_value:.3g}')
ax.set_ylabel('Densité référentielle')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

############################################################################################
# CE1 vs CE2 - Italien
gruppo1 = df2[df2['group'] == 'CE1']['density']
gruppo2 = df2[df2['group'] == 'CE2']['density']
statistica, p_value = ttest_ind(gruppo1, gruppo2, equal_var=False)
print(f'Statistica t: {statistica}')
print(f'P-value: {p_value}')

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([gruppo1, gruppo2], labels=['CE1', 'CE2'], patch_artist=True)
annotate_boxplot(ax, [gruppo1.values, gruppo2.values], bp)

ax.set_title(f'Comparaison densité référentielle entre CE1 et CE2 italiens\nTest t: p = {p_value:.3g}')
ax.set_ylabel('Densité référentielle')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#############################################################################################
# Confronto CE1 vs CE2 per entrambe le lingue
gruppo1_fr = df1[df1['group'] == 'CE1']['density']
gruppo2_fr = df1[df1['group'] == 'CE2']['density']
stat_fr, p_fr = ttest_ind(gruppo1_fr, gruppo2_fr, equal_var=True)

gruppo1_it = df2[df2['group'] == 'CE1']['density']
gruppo2_it = df2[df2['group'] == 'CE2']['density']
stat_it, p_it = ttest_ind(gruppo1_it, gruppo2_it, equal_var=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bp1 = ax1.boxplot([gruppo1_fr, gruppo2_fr], labels=['CE1', 'CE2'], patch_artist=True)
annotate_boxplot(ax1, [gruppo1_fr.values, gruppo2_fr.values], bp1)
ax1.set_title(f'Densité référentielle - Français\nTest t: p = {p_fr:.3g}')
ax1.set_ylabel('Densité référentielle')
ax1.grid(True, linestyle='--', alpha=0.6)

bp2 = ax2.boxplot([gruppo1_it, gruppo2_it], labels=['CE1', 'CE2'], patch_artist=True)
annotate_boxplot(ax2, [gruppo1_it.values, gruppo2_it.values], bp2)
ax2.set_title(f'Densité référentielle - Italien\nTest t: p = {p_it:.3g}')
ax2.set_ylabel('Densité référentielle')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

#############################################################################################
# Comparer chaque niveau entre langues
gruppo1_fr = df1[df1['group'] == 'CE1']['density']
gruppo1_it = df2[df2['group'] == 'CE1']['density']
stat1, p1 = ttest_ind(gruppo1_fr, gruppo1_it, equal_var=False)

gruppo2_fr = df1[df1['group'] == 'CE2']['density']
gruppo2_it = df2[df2['group'] == 'CE2']['density']
stat2, p2 = ttest_ind(gruppo2_fr, gruppo2_it, equal_var=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bp1 = ax1.boxplot([gruppo1_fr, gruppo1_it], labels=['français', 'italien'], patch_artist=True)
annotate_boxplot(ax1, [gruppo1_fr.values, gruppo1_it.values], bp1)
ax1.set_title(f'Densité référentielle - CE1\nTest t: p = {p1:.3g}')
ax1.set_ylabel('Densité référentielle')
ax1.grid(True, linestyle='--', alpha=0.6)

bp2 = ax2.boxplot([gruppo2_fr, gruppo2_it], labels=['français', 'italien'], patch_artist=True)
annotate_boxplot(ax2, [gruppo2_fr.values, gruppo2_it.values], bp2)
ax2.set_title(f'Densité référentielle - CE2\nTest t: p = {p2:.3g}')
ax2.set_ylabel('Densité référentielle')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

