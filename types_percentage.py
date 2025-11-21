# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:57:49 2025

@author: matil

mesure percentage of mention types for each chain in text
input - excel types mentions
output - le sequenze più frequenti 

text | character/s | SNdef | SNindef | PRON | PROPN ....

use manually wrangled data as an input
v1
C:/Users/matil/Documents/decorscol/stats_TALN_2025/typologie_mentions_fr_V4.xlsx
C:/Users/matil/Documents/decorscol/stats_TALN_2025/ita_types_v2.xlsx
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    combined_df.sort_values(by=['Source', 'tag'], inplace=True)

    return combined_df

def calculate_sheetname_percentages(df):
    # Count total rows for each Source/tag
    group_counts = df.groupby(['Source', 'tag']).size().reset_index(name='Total')

    # Count SheetName occurrences per Source/tag
    counts = df.groupby(['Source', 'tag', 'SheetName']).size().reset_index(name='Count')

    # Merge to calculate percentage
    merged = pd.merge(counts, group_counts, on=['Source', 'tag'])
    merged['Percentage'] = (merged['Count'] / merged['Total']) * 100

    # Pivot to get desired format
    pivot_df = merged.pivot_table(index=['Source', 'tag'],
                                  columns='SheetName',
                                  values='Percentage',
                                  fill_value=0).reset_index()

    return pivot_df

def plot_sheetname_distribution(pivot_df):
    # Melt to long-form for Seaborn plotting
    melted = pd.melt(pivot_df, id_vars=['Source', 'tag'],
                     var_name='SheetName', value_name='Percentage')

    # Create a combined key for plotting
    melted['Source_tag'] = melted['Source'].astype(str) + ' / ' + melted['tag'].astype(str)




# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent


# Load the Excel file
file_path_fr = project_dir/"sheets/francais/typologie_mentions_fr_V4.xlsx"
file_path_it = project_dir/"sheets/italien/ita_types_v2.xlsx"



# === Run the complete process ===
file_path = file_path_fr  # Replace this with your actual Excel file path

# Load, process, compute, and visualize
combined_df = load_and_process_excel(file_path)
pivot_df = calculate_sheetname_percentages(combined_df)

# 🔹 Aggiungi qui il calcolo del massimo tag_occurrences per ogni Source/Tag
max_tag_occurrences = (
    combined_df.groupby(['Source', 'tag'])['tag_occurrences']
    .max()
    .reset_index(name='Max_Tag_Occurrences')
)

# 🔹 Unisci al pivot_df
pivot_df = pd.merge(pivot_df, max_tag_occurrences, on=['Source', 'tag'], how='left')

plot_sheetname_distribution(pivot_df)

# Optionally print the pivoted dataframe
print(pivot_df)
pivot_df = pivot_df[~pivot_df['Max_Tag_Occurrences'].isin([1, 2])]

# # Seleziona solo i campi necessari
# dist_df = pivot_df[['Source', 'tag', 'Max_Tag_Occurrences']].copy()
# dist_df['Source_Tag'] = dist_df['Source'].astype(str) + ' / ' + dist_df['tag'].astype(str)

print(list(pivot_df.columns))

##raggruppare per catene simili
# Le colonne da usare per trovare righe "uguali"
group_columns = ['autre', 'det_poss', 'numerals', 'pron', 
                 'propn', 'sn_def', 'sn_dem', 'sn_indef', 
                 'sn_no_det', 'sn_poss', 'verb']



####creare i profili di ogni catena 
# Raggruppa per Source e Value, e ordina per tag_occurrences crescente
sorted_df = combined_df.sort_values(by=['Source', 'tag', 'tag_occurrences'], ascending=[True, True, True])

max_occurrences = (
    sorted_df.groupby(['Source', 'tag'])['tag_occurrences']
    .max()
    .reset_index(name='Max_Tag_Occurrences')
)

# Ordina prima
sorted_df = combined_df.sort_values(by=['Source', 'tag', 'tag_occurrences'])

# Aggrega i SheetName ordinati in una lista
sheetname_ordered = (
    sorted_df.groupby(['Source', 'tag'])['SheetName']
    .apply(list)
    .reset_index(name='OrderedSheetNames')
)

pivot_df2 = pd.merge(max_occurrences, sheetname_ordered, on=['Source', 'tag'])


# 1. Espandi i valori Ordered_SheetNames su colonne separate
sheetname_expanded = pd.DataFrame(
    pivot_df2['OrderedSheetNames'].to_list(),
    columns=[f'Sheet_{i+1}' for i in range(pivot_df2['OrderedSheetNames'].apply(len).max())]
)



# 2. Combina con le altre colonne (Source, Tag, Max_Tag_Occurrences)
export_df = pd.concat(
    [pivot_df2[['Source', 'tag', 'Max_Tag_Occurrences']], sheetname_expanded],
    axis=1
)


# 3. Esporta su Excel
export_df.to_excel(project_dir/"sheets/francais/chains_profiles_fr.xlsx", index=False)

# Seleziona le prime 5 colonne Sheet_*
sheet_columns = [col for col in export_df.columns if col.startswith('Sheet_')][0:3]

# Calcola le frequenze delle combinazioni
combo_counts = export_df[sheet_columns].value_counts().reset_index(name='Frequency')

# Ordina dalla combinazione più frequente a meno frequente
combo_counts = combo_counts.sort_values(by='Frequency', ascending=False)

# Mostra le top 10 combinazioni
print(combo_counts.head(10))

# Combina i nomi delle Sheet in una stringa per asse X
combo_counts['Combo'] = combo_counts[sheet_columns].astype(str).agg(' - '.join, axis=1)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=combo_counts.head(10), x='Frequency', y='Combo', palette='pastel')
plt.title('Top 10 combinazioni di Sheet più frequenti')
plt.xlabel('Frequenza')
plt.ylabel('Sequenza Sheet')
plt.tight_layout()
plt.show()

