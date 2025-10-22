# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:59:24 2025

@author: matil
calcul de la densité référentielle
prends en input le fichier csv contenant chaque mention et ses caracteristiques
elimine du decompte les mentions imbriquées dans des mentions plus larges 

"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get this script's folder (codes/)
script_dir = Path(__file__).parent

# Go up one level to project/
project_dir = script_dir.parent
data_dir = project_dir / "sheets/francais"

#row x text == nb mentions
# nb text | nb mentions | nb tokens depuis autre tableau 

##combien de tokens dans les mentions sum tot et sum par niveau
def remove_contained_spans(df):
    kept_spans = []
    removed_spans = []

    # Group by 'Source' to process spans within the same source separately
    for source, group in df.groupby('Source'):
        # Sort spans within each group
        group_sorted = group.sort_values(by=['begin', 'end'], ascending=[True, False]).reset_index(drop=True)

        # Track kept spans and max end value
        filtered_spans = []
        max_end = -1
        
        for _, row in group_sorted.iterrows():
            begin, end = row['begin'], row['end']

            # If the span is contained within a larger span, remove it
            if end <= max_end:
                removed_spans.append(row)
            else:
                filtered_spans.append(row)
                max_end = max(max_end, end)  # Update max_end
        
        kept_spans.extend(filtered_spans)

    # Convert to DataFrame
    df_filtered = pd.DataFrame(kept_spans)
    df_removed = pd.DataFrame(removed_spans)

    # Save removed spans to CSV
    df_removed.to_csv(data_dir/'removed_spans.csv', index=False)

    return df_filtered  # Return the cleaned DataFrame



#CALCUL DE LA DENSITE référentielle : nb tokens mentions / nb tokens texte 
def calcul_density(fr, toks_df, filename) : 
    # Filter out empty values in 'Text' column
    df_fr_filtered = fr[fr['Source'] != '']

    # Group by 'Text' and count occurrences
    df_counts = df_fr_filtered.groupby('Source').size().reset_index(name='Mentions')
    df_counts['Source'] = df_counts['Source'].str[:-4]

    #mentions_sum = df_counts['Mentions'].sum()

    # Rename 'old_name' to 'new_name'
    df_counts.rename(columns={'Source': 'texte'}, inplace=True)
    #print(df_counts)

    fr_toks = pd.read_csv(toks_df)
    fr_toks = fr_toks[['texte', 'nbTokNoPunct']]
    #corpus_francais_v3
    # Merge on the 'text' column
    df_merged = df_counts.merge(fr_toks, on='texte', how='inner')  # 'inner' ensures only matching "text" values remain

    print(df_merged)

    #densité = ratio nb ER / nb tokens (no punct)
    df_merged['density'] = df_merged['Mentions'] / df_merged['nbTokNoPunct']*100
    df_merged["niveau"] = df_merged["texte"].str.extract(r"(CE1|CE2)")
    df_merged.to_csv(filename, index=False)

    df_CE1 = df_merged[df_merged['texte'].str.contains('CE1', case=False, na=False)]
    df_CE2 = df_merged[df_merged['texte'].str.contains(('CE2'), case=False, na=False)]

    moy_density = df_merged['density'].mean()
    
    return df_merged, moy_density, df_CE1, df_CE2

###MAIN FR
fr = pd.read_csv(data_dir / "toutes_mentions_corpus_detail_interdistance_fr.csv")
fr = fr.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

toks_df = data_dir / "corpus_francais_v3.csv"
res_fr = data_dir / "density_fr.csv"
fr_filtered = remove_contained_spans(fr)

df_merged, moy_density, df_CE1, df_CE2 = calcul_density(fr_filtered, toks_df, res_fr)

print("*** FRANCAIS ***")
print("Densité moyenne : ", moy_density)
print("Densité min : ", df_merged['density'].min())
print("Densité max : ", df_merged['density'].max())
print("Ecart type : ", df_merged['density'].std())

print("Densité moyenne CE1 : ", df_CE1['density'].mean())
print("Densité min CE1 : ", df_CE1['density'].min())
print("Densité max  CE1 : ", df_CE1['density'].max())
print("Ecart type : ", df_CE1['density'].std())


print("Densité moyenne CE2 : ", df_CE2['density'].mean())
print("Densité min CE2: ", df_CE2['density'].min())
print("Densité max CE2: ", df_CE2['density'].max())
print("Ecart type : ", df_CE2['density'].std())

####istogramma
sns.set_palette("pastel")
data=df_merged['density']
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data,
    bins=30,
    color=sns.color_palette("pastel")[0],
    alpha=0.6
)

# Add labels above each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')
ax.set_title('Distribution de la densité par texte - corpus français')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)




# Plot KDE scaled to counts
# from scipy.stats import gaussian_kde

# kde = gaussian_kde(data)
# x_vals = np.linspace(min(data), max(data), 200)
# # Scale KDE by total count and bin width
# bin_width = (max(data) - min(data)) / 30
# scaled_kde = kde(x_vals) * len(data) * bin_width

# ax.plot(x_vals, scaled_kde, color=sns.color_palette("pastel")[1], linewidth=2)

# plt.xlabel("Densité référentielle", fontsize=14)
# plt.ylabel("Nombre de textes", fontsize=14)
# plt.tight_layout()
# plt.show()


# ####MAIN ITA
#TODO change to data_dir 
data_dir = project_dir / "sheets/italien"

ita = pd.read_csv(data_dir/"toutes_mentions_corpus_detail_interdistance_ita.csv")
ita = ita.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

toks_df = data_dir/"corpus_italien.csv"
res_ita = data_dir/"density_ita_v2.csv"
ita_filtered = remove_contained_spans(ita)

df_ita, moy_density, df_CE1_it, df_CE2_it = calcul_density(ita_filtered, toks_df, res_ita)

print("*** ITA ***")
print("Densité moyenne : ", moy_density)
print("Densité min : ", df_ita['density'].min())
print("Densité max : ", df_ita['density'].max())
print("Ecart type : ", df_ita['density'].std())

print("Densité moyenne CE1 : ", df_CE1_it['density'].mean())
print("Densité min CE1 : ", df_CE1_it['density'].min())
print("Densité max  CE1 : ", df_CE1_it['density'].max())
print("Ecart type : ", df_CE1_it['density'].std())


print("Densité moyenne CE2 : ", df_CE2_it['density'].mean())
print("Densité min CE2: ", df_CE2_it['density'].min())
print("Densité max CE2: ", df_CE2_it['density'].max())
print("Ecart type : ", df_CE2_it['density'].std())





"""
# print("*** ITA ***")
# print("Densité moyenne : ", moy_density)
# print("Densité moyenne CE1 : ", moy_density_CE1)
# print("Densité moyenne CE2 : ", moy_density_CE2)
italian = pd.read_csv('annotations_italien/density_ita_v2.csv')
data = italian['density']

####istogramma
sns.set_palette("pastel")
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data,
    bins=30,
    color=sns.color_palette("pastel")[0],
    alpha=0.6
)

# Add labels above each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')

# Plot KDE scaled to counts
from scipy.stats import gaussian_kde

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 200)
# Scale KDE by total count and bin width
bin_width = (max(data) - min(data)) / 30
scaled_kde = kde(x_vals) * len(data) * bin_width

ax.plot(x_vals, scaled_kde, color=sns.color_palette("pastel")[1], linewidth=2)

plt.xlabel("Densité référentielle", fontsize=14)
plt.ylabel("Nombre de textes", fontsize=14)
plt.tight_layout()
plt.show()

"""
