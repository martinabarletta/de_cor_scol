# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:21:28 2025

@author: matil

code pour le calcul du coefficient de perret (2000)
stabilité, instabilité et coefficient oberlé CNSL

"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent

def stability(file):
    df = pd.read_csv(file)
    
    # Count occurrences of each tag in each Source
    tag_counts = df.groupby(['tag', 'Source']).size().reset_index(name='count')
    
    # Keep only (tag, Source) combinations that appear more than twice
    valid_pairs = tag_counts[tag_counts['count'] > 2][['tag', 'Source']]
    
    # Merge to keep only rows in df that match valid (tag, Source) pairs
    filtered_df = df.merge(valid_pairs, on=['tag', 'Source'])
    
    # Step 1: Compute the max 'tag_occurrences' for each (tag, Source) in the original df
    max_occurrences = df.groupby(['tag', 'Source'])['tag_occurrences'].max().reset_index()
    max_occurrences = max_occurrences.rename(columns={'tag_occurrences': 'max_tag_occurrences'})
    
    # Step 2: Merge this info into your filtered dataframe
    filtered_df = filtered_df.merge(max_occurrences, on=['tag', 'Source'], how='left')
    
    # Filtering rows where 'Text' column contains "NOUN" --> mentions nominales (SN)
    sn_mentions = filtered_df[filtered_df['POSno'].str.contains('NOUN', na=False)]
    #lower case to avoid upper/lower case bias
    sn_mentions['mention'] = sn_mentions['mention'].str.lower()
    
    ###calcul du vrai coefficient de Perret : 
    #si la première mention de la chaine est un SN indéfini, exclure du décompte
    #group mentions by tag
    #if tag_occurrences is 1 and first POSno is DET and 1st part of morphNoPunct is equal to Definite=Ind
    
    # sn_indef = sn_mentions[
    #     (sn_mentions['tag_occurrences'] == 1) &
    #     (sn_mentions['POSno'].str.startswith("['DET")) &
    #     (sn_mentions['morphNoPunct'].str.startswith("['Definite=Ind"))
    # ]
    # print(sn_indef)
    
    # sn indef non viene usato ma serve per controllare che siano state escluse le menzioni mirate
    # mentions filtrées : seulement mentions nominales, exclus les première mentions de chaine SN indéfini
    sn_no_indef = sn_mentions[
        ~(
          (sn_mentions['tag_occurrences'] == 1) &
          (sn_mentions['POSno'].str.startswith("['DET'")) &
          (sn_mentions['morphNoPunct'].str.startswith("['Definite=Ind"))
         )
    ]
    
    #calcul du coefficient
    result = sn_no_indef.groupby(['Source', 'tag']).agg(
        nominalMent=('mention', 'count'),  # Count total mentions
        unique=('mention', pd.Series.nunique),  # Count unique mentions
        len_chaine=('max_tag_occurrences', 'max')  # Keep the max per (Source, tag)
    ).reset_index()
    
    
    # Add the new column with the formula nb mentions nominales / nb mentions uniques
    # * 100 --> instabilité (Rousier-Vercruyssen et Landragin)
    # CNSL = indice normalisé
    result['stability'] = result['nominalMent'] / result['unique']
    result['instability'] = result['unique'] / result['nominalMent'] * 100
    #1 - ((x-1) / (n-1))
    #x = mentions uniques, n nb total de mentions nominales
    result['CNSL'] = 1 - ((result['unique']-1) / (result['nominalMent']-1))
    result['CNSL'] = result['CNSL'].fillna(1)

    #tableau des taux pour chaque texte fr
    result.to_csv(project_dir/'instability.csv', index=False)
    
    #tableau recap pour manuscrit
    header=['chaines', 'stability', 'instability', 'CNSL']

    # Calcul des statistiques voulues
    recap = {
        'chaines': len(result),        # moyenne
        'stability': result['stability'].mean(),    # moyenne
        'instability': result['instability'].mean(),# moyenne
        'CNSL': result['CNSL'].mean()                # somme
    }
    
    # --- Filtrage pour CE1 et CE2
    df_CE1 = result[result['Source'].str.contains('CE1', case=False, na=False)]
    df_CE2 = result[result['Source'].str.contains('CE2', case=False, na=False)]
    
    # --- Calcul des stats pour CE1
    recap_CE1 = {
        'chaines': len(df_CE1),
        'stability': df_CE1['stability'].mean(),
        'instability': df_CE1['instability'].mean(),
        'CNSL': df_CE1['CNSL'].mean()
    }
    
    # --- Calcul des stats pour CE2
    recap_CE2 = {
        'chaines': len(df_CE2),
        'stability': df_CE2['stability'].mean(),
        'instability': df_CE2['instability'].mean(),
        'CNSL': df_CE2['CNSL'].mean()
    }
    
    # --- Création du tableau récapitulatif final
    tableau_recap = pd.DataFrame([recap, recap_CE1, recap_CE2], columns=header, index=['Global', 'CE1', 'CE2'])
    
    print(tableau_recap)  
    
    return result, tableau_recap



##français
file_path_fr = project_dir/"sheets/francais/toutes_mentions_corpus_detail_interdistance_fr.csv"
result, tableau_recap_fr = stability(file_path_fr)
result.to_csv(project_dir/"sheets/francais/results_fr.csv")
tableau_recap_fr.to_csv(project_dir/"sheets/francais/recap_stability_fr.csv")

###italien
file_path_it = project_dir/"sheets/italien/toutes_mentions_corpus_detail_interdistance_ita.csv"
df_it, tableau_recap = stability(file_path_it)


###plots
def plot_cnsl(result):
    # Replace 'your_column' with your actual column name
    column_name = 'CNSL'
    total_nans = result[column_name].isna().sum()
    print(f"Number of NaN values dropped: {total_nans}")
    
    # Drop NA values for plotting
    data = result[column_name].dropna()
    
    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Histogram
    axs[0].hist(data, bins=20, edgecolor='black')
    axs[0].set_title('Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    
    # 2. KDE Plot
    sns.kdeplot(data, ax=axs[1], bw_adjust=0.3)
    axs[1].set_title('KDE Plot')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Density')
    
    # 3. Box Plot
    axs[2].boxplot(data, vert=False)
    axs[2].set_title('Box Plot')
    axs[2].set_xlabel('Value')
    
    # 4. Violin Plot
    sns.violinplot(x=data, ax=axs[3])
    axs[3].set_title('Violin Plot')
    axs[3].set_xlabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    # Create KDE values
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE curve
    kde = sns.kdeplot(data, bw_adjust=0.3, ax=ax, color='blue', linewidth=2)
    
    # Get the x and y values from the KDE line
    x_vals = kde.get_lines()[0].get_xdata()
    y_vals = kde.get_lines()[0].get_ydata()
    
    # Add value labels (transparent, only some points to avoid clutter)
    for i in range(0, len(x_vals), max(1, len(x_vals)//20)):
        x = x_vals[i]
        y = y_vals[i]
        ax.text(x, y + 0.002, f'{y:.2f}', ha='center', fontsize=8, alpha=0.5)
    
    # Final plot formatting
    ax.set_title('', fontsize=14)
    ax.set_xlabel('CNSL', fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with density=True normalizes the area to 1
    counts, bins, patches = ax.hist(data, bins=20, edgecolor='black', density=False)
    
    # Add value labels on top of bars
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height + 0.5,
                    f'{int(height)}', ha='center', fontsize=12, alpha=0.6)
    
    # Labels and title
    ax.set_title('', fontsize=14)
    ax.set_xlabel('CNSL', fontsize=12)
    ax.set_ylabel('Nombre des chaines', fontsize=12)
    plt.tight_layout()
    plt.show()

plot_cnsl(result)
plot_cnsl(df_it)
