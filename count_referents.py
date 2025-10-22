# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:49:33 2025

@author: matil
"""
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


# Get this script's folder (codes/)
script_dir = Path(__file__).parent


# Go up one level to project/
project_dir = script_dir.parent
data_dir = project_dir / "sheets/italien"

def count_positive_values(file_path):
    # Load CSV
    df = pd.read_csv(file_path)

    # Select columns from index 2 (3rd column) to the end
    target_columns = df.columns[4:]

    # Count how many values > 0 in those columns per row
    df['nb_referents'] = (df[target_columns] > 0).sum(axis=1)

    # Print results
    result_df = df[[df.columns[1], 'nb_referents']]

    return result_df

df_result = count_positive_values(data_dir / "referents_corpus_italien.csv")
df_result["langue"] = "italien"
df_result["niveau"] = df_result["texte"].str.extract(r"(CE1|CE2)")
df_result.to_csv(data_dir/"referents_fr.csv", index=False)

media = df_result['nb_referents'].mean()

sns.violinplot(x='niveau', y='nb_referents', data=df_result, inner='box', palette='pastel')

# Calcul des statistiques
stats = df_result.groupby('niveau')['nb_referents'].agg(['mean', 'max', 'min']).reset_index()

# Ajout des étiquettes
for i, row in stats.iterrows():
    # Position x = index du niveau, y = médiane
    plt.text(i, row['max'], f"Max = {row['max']:.1f}", 
             ha='center', va='top', color='black', fontweight='bold')
    plt.text(i, row['mean'], f"Moy = {row['mean']:.1f}", 
             ha='center', va='top', color='black', fontstyle='italic')
    plt.text(i, row['min'], f"Min = {row['min']:.1f}", 
             ha='center', va='bottom', color='black', fontweight='bold')

plt.title("Distribution du nombre de référents par niveau scolaire - corpus français")
plt.xlabel("Niveau scolaire")
plt.ylabel("Nombre de référents par texte")
plt.tight_layout()
plt.show()

##barplot
plt.figure(figsize=(7,5))
# Countplot
ax = sns.countplot(
    data=df_result,
    x='nb_referents',
    hue='niveau',
    palette='pastel'
)

# Ajout des étiquettes sur les barres
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # éviter les barres vides
        ax.text(
            p.get_x() + p.get_width()/2,   # position horizontale
            height + 0.05,                 # position verticale (un peu au-dessus)
            int(height),                   # valeur à afficher
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )


plt.title("Nombre de textes selon le nombre de référents et le niveau scolaire - corpus italien")
plt.xlabel("Nombre de référents dans le texte")
plt.ylabel("Nombre de textes")
plt.legend(title="Niveau scolaire")
plt.tight_layout()
plt.show()


#comment répresenter la distribution des référents par niveaux scolaires ?
##français






