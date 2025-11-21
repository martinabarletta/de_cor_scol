# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:45:27 2025

@author: matil
heatmap per frequenza categorie secondo posizione nella catena
prends en entrée le fichier excel (déjà nettoyé), un sheet par type de mention que nous avons crée 
de manière semi-automatique (regroupement automatique par pattern et nettoyage manuel)
le fichier excel en entrée est crée par le code mention_type_alternative_ita ou mention_type_alternative

"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define a mapping of sheet names to group names
sheet_groups = {
    "sn_def": "SN défini",
    "sn_indef": "SN indéfini",
    "verb": "Anaphore Zéro",
    "pron": "Pronom",
    "propn": "Nom Propre", 
    "det_poss": "SN possessif", 
    "sn_poss": "SN possessif",
    "sn_dem" : "SN démonstratif",
    "sn_no_det" : "SN sans déterminant",
    "numerals" : "Autre",
    "autre" : 'Autre'
}


def calculate_ERType(file_path, sheet_groups) :
    xls = pd.ExcelFile(file_path)
    
    # Dictionary to store dataframes
    dfs = {}
    
    # Read each sheet into a dataframe
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Ensure 'Index' column exists
        if 'Unnamed: 0' in df.columns:
            dfs[sheet_name] = df  # Store dataframe
    
    total_rows = sum(len(df.dropna(how="all")) for df in dfs.values())
    print(f"Total number of non-empty rows across all sheets: {total_rows}")
    
    
    # Create a list to store sheet names, groups, and row counts
    sheet_info = []
    
    # Iterate through each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Drop completely empty rows
        df = df.dropna(how="all")
        
        # Count rows excluding the header
        row_count = len(df)
        
        # Determine the group (default to "Uncategorized" if not found in mapping)
        group_name = sheet_groups.get(sheet_name, "Uncategorized")
        
        # Append sheet name, group, and row count to the list
        sheet_info.append({"Sheet Name": sheet_name, "Type": group_name, "Nb": row_count})
    
    # Convert to DataFrame
    sheet_info_df = pd.DataFrame(sheet_info)

    
    # Now, group by 'Group' and sum the 'Row Count'
    grouped_df = sheet_info_df.groupby("Type")["Nb"].sum().reset_index()
    
    # Display the grouped DataFrame
    print(f" grouped df from file path {file_path}\n",grouped_df)
    return grouped_df, dfs


def filter_anaphoras(dfs_dict, filtered_df):
    # Define the columns used for matching
    comparison_columns = ['begin', 'end', 'mention', 'tag', 'tag_occurrences', 'Source']

    # Clean filtered_df
    filtered_df_selected = filtered_df[comparison_columns].copy()

    for col in comparison_columns:
        if col in filtered_df_selected.columns:
            if filtered_df_selected[col].dtype == "object":
                filtered_df_selected[col] = filtered_df_selected[col].astype(str).str.strip()
            else:
                filtered_df_selected[col] = pd.to_numeric(filtered_df_selected[col], errors="coerce")

    # Container for all removed rows
    dropped_rows_list = []

    # Process each dataframe
    for key in dfs_dict:
        df = dfs_dict[key]
        existing_columns = [c for c in comparison_columns if c in df.columns]

        # Clean types in the target dataframe
        for col in existing_columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Merge to detect which rows will be dropped
        merged = df.merge(
            filtered_df_selected[existing_columns],
            on=existing_columns,
            how="left",
            indicator=True
        )

        # Save the rows that will be removed
        dropped_rows = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        dropped_rows["origin"] = key  # keep track of origin
        dropped_rows_list.append(dropped_rows)

        # Keep only rows NOT matched
        dfs_dict[key] = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    # Concatenate all removed rows in one dataframe (same format as originals)
    if dropped_rows_list:
        dropped_df = pd.concat(dropped_rows_list, ignore_index=True)
    else:
        dropped_df = pd.DataFrame(columns=comparison_columns + ["origin"])

    return dfs_dict, dropped_df


def separate_anaphoras_and_singletons(dropped_df):
    """
    Separate anaphoras and singletons from dropped_df based on:
    - same Source
    - same tag
    - anaphora = exactly two rows: occurrences 1 and 2
    - singleton = only one row with occurrences == 1
    """

    # group by Source and tag
    grouped = dropped_df.groupby(['Source', 'tag'])

    anaphoras = []
    singletons = []

    for (source, tag), group in grouped:
        occurrences = list(group['tag_occurrences'])

        # --- case 1: anaphora ---
        # must have exactly two rows, and occurrences must be {1, 2}
        if sorted(occurrences) == [1, 2] and len(group) == 2:
            anaphoras.append(group)

        # --- case 2: singleton ---
        # must have only 1 row, and tag_occurrences == 1
        elif len(group) == 1 and occurrences == [1]:
            singletons.append(group)

        # anything else is ignored (e.g. multiple anaphoras, malformed cases)

    # concat results into single dataframes
    anaphoras_df = pd.concat(anaphoras, ignore_index=True) if anaphoras else pd.DataFrame()
    singletons_df = pd.concat(singletons, ignore_index=True) if singletons else pd.DataFrame()

    return anaphoras_df, singletons_df


#######################################
#                MAIN                 #
#######################################
# Get this script's folder (codes/)
script_dir = Path(__file__).parent
# Go up one level to project/
project_dir = script_dir.parent


# Load the Excel file
file_path_fr = project_dir/"sheets/francais/typologie_mentions_fr_V4.xlsx"
file_path_it = project_dir/"sheets/italien/ita_types_v2.xlsx"

#toutes mentions
french, fr_dfs = calculate_ERType(file_path_fr, sheet_groups)
italian, it_dfs = calculate_ERType(file_path_it, sheet_groups)

#####filter out singletons and anaphoras##############################################################################
filtered_df_fr = pd.read_csv(project_dir/"sheets/francais/anaphores_fr.csv") #<- fatto con interdistance_table
filtered_df_it = pd.read_csv(project_dir/"sheets/italien/anaphores_it.csv") #<- fatto con interdistance_table

filtered_fr, dropped_fr = filter_anaphoras(fr_dfs, filtered_df_fr)
filtered_ita, dropped_ita = filter_anaphoras(it_dfs, filtered_df_it)

anaphoras_fr, singletons_fr = separate_anaphoras_and_singletons(dropped_fr)
anaphoras_it, singletons_it = separate_anaphoras_and_singletons(dropped_ita)

#TODO find a way to separate anaphoras from singletons
def convert_and_count_categories(df, category_column, sheet_groups):
    """
    Map categories in `category_column` using `sheet_groups`
    and count how many rows fall into each mapped category.
    """

    # Convert to str to avoid type issues
    df = df.copy()
    df[category_column] = df[category_column].astype(str).str.strip()

    # Map categories — unknown labels become 'Autre'
    df['Category_mapped'] = df[category_column].map(sheet_groups).fillna('Autre')

    # Count categories
    counts = df['Category_mapped'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']

    return df, counts

singletons_fr, counts_sing_fr = convert_and_count_categories(singletons_fr, 'origin', sheet_groups)
singletons_it, counts_sing_it = convert_and_count_categories(singletons_it, 'origin', sheet_groups)


def convert_and_count_categories_anaphore(df, category_column, sheet_groups):
    """
    Map categories in `category_column` using `sheet_groups`
    and produce:
    - total counts per category
    - counts for tag_occurrences == 1
    - counts for tag_occurrences == 2
    """

    df = df.copy()

    # Ensure the category column is clean
    df[category_column] = df[category_column].astype(str).str.strip()

    # Apply mapping
    df['Category_mapped'] = df[category_column].map(sheet_groups).fillna('Autre')

    # ---- TOTAL counts ----
    counts_total = (
        df['Category_mapped']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Category', 'Category_mapped': 'Count'})
    )

    # ---- Counts for tag_occurrences == 1 ----
    df_occ1 = df[df['tag_occurrences'] == 1]
    counts_occ1 = (
        df_occ1['Category_mapped']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Category', 'Category_mapped': 'Count'})
    )

    # ---- Counts for tag_occurrences == 2 ----
    df_occ2 = df[df['tag_occurrences'] == 2]
    counts_occ2 = (
        df_occ2['Category_mapped']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Category', 'Category_mapped': 'Count'})
    )

    return df, counts_total, counts_occ1, counts_occ2

anaphore_fr, counts_total_anaf_fr, counts_occ1, counts_occ2 = convert_and_count_categories_anaphore(anaphoras_fr, 'origin', sheet_groups)
tot_counts_occ1 = counts_occ1['Count'].sum()
tot_counts_occ2 = counts_occ2['Count'].sum()


anaphore_it, counts_total_anaf_it, counts_occ1_it, counts_occ2_it = convert_and_count_categories_anaphore(anaphoras_it, 'origin', sheet_groups)


# Define the new function to work with a dictionary of DataFrames
def calculate_ERType_from_dict(dfs_dict, sheet_groups, mention_order = None):
    # Dictionary to store the information about rows for each dataframe
    sheet_info = []
    
    # Total number of non-empty rows across all dataframes
    total_rows = sum(len(df.dropna(how="all")) for df in dfs_dict.values())
    print(f"Total number of non-empty rows across all dataframes: {total_rows}")
    
    # Iterate through each dataframe in the dictionary
    for sheet_name, df in dfs_dict.items():
        # Drop completely empty rows
        df_cleaned = df.dropna(how="all")
        
        # Apply the filter if filter_tag_value is provided
        if mention_order is not None and 'tag_occurrences' in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned['tag_occurrences'] == mention_order]
            
        # Count rows excluding the header
        row_count = len(df_cleaned)
        
        # Determine the group (default to "Uncategorized" if not found in mapping)
        group_name = sheet_groups.get(sheet_name, "Uncategorized")
        
        # Append dataframe (sheet_name), group, and row count to the list
        sheet_info.append({"Sheet Name": sheet_name, "Type": group_name, "Nb": row_count})
    
    # Convert the sheet_info list to a DataFrame
    sheet_info_df = pd.DataFrame(sheet_info)
    
    # Now, group by 'Type' (group name) and sum the 'Nb' (row count)
    grouped_df = sheet_info_df.groupby("Type")["Nb"].sum().reset_index()
    
    # Display the grouped DataFrame
    print(grouped_df)
    
    return grouped_df

#seulement chaines
# Call the modified function
french = calculate_ERType_from_dict(filtered_fr, sheet_groups)
italian = calculate_ERType_from_dict(filtered_ita, sheet_groups)

#position 1 dans seulement chaines 
french_1 = calculate_ERType_from_dict(filtered_fr, sheet_groups, 1)
italian_1 = calculate_ERType_from_dict(filtered_ita, sheet_groups, 1)

#position 1 dans seulement chaines 
french_2 = calculate_ERType_from_dict(filtered_fr, sheet_groups, 2)
italian_2 = calculate_ERType_from_dict(filtered_ita, sheet_groups, 2)


#######################################
#                PLOTS                #
#######################################
def plot_barres(french, italian):
    #plot similarities and differences between the two languages in terms of distribution : percentage representation
    merged_df = pd.merge(french, italian, on='Type', suffixes=('_fr', '_it'))
    
    # Step 2: Calculate the percentage for each column
    merged_df['% fr'] = merged_df['Nb_fr'] / merged_df['Nb_fr'].sum() * 100
    merged_df['% it'] = merged_df['Nb_it'] / merged_df['Nb_it'].sum() * 100
    
    # Step 1: Add a row with the sum of each column
    sum_row = merged_df[['Nb_fr', 'Nb_it']].sum()
    sum_row['Type'] = 'Total'  # Add the 'Total' label for the Category
    merged_df = merged_df.append(sum_row, ignore_index=True)
    
    
    # # Plotting the merged DataFrame
    # merged_df.plot(x='Type', y=['% fr', '% it'], kind='bar')
    # plt.title('Comparison of Values by Category')
    # plt.ylabel('Values')
    # plt.show()
    # Use Seaborn pastel color palette
    pastel_colors = sns.color_palette("pastel")  # Get 2 pastel colors for the bars
    pastel_colors = pastel_colors[3:5]
    
    #fixed categories for x axis label order
    categories = ["Pronom", "SN défini", "SN indéfini",
                  "SN possessif", "SN démonstratif", "SN sans déterminant",
                  "Nom Propre", "Anaphore Zéro", "Autre"]
    
    
    # Assuming 'merged_df' is your DataFrame and contains the columns '% fr' and '% it'
    # Exclude the "Total" row
    df_plot = merged_df[merged_df['Type'] != 'Total']
    
    df_plot['Type'] = pd.Categorical(df_plot['Type'], categories=categories, ordered=True)
    
    # Sort the dataframe to match the category order
    df_plot = df_plot.sort_values('Type')
    
    
    # Create the bar plot with custom width and spacing between groups
    ax = df_plot.plot(x='Type', y=['% fr', '% it'], kind='bar', width=0.8, figsize=(12, 8), 
                      color=pastel_colors, edgecolor='black')
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels above each bar, centered over each bar
    for i in range(len(df_plot)):
        # Center the text over the bars (adjusting the x position to the center of the bar)
        ax.text(i - 0.2, df_plot['% fr'].iloc[i] + 0.5, f"{df_plot['% fr'].iloc[i]:.2f}", color='black', ha='center', fontsize=14)
        ax.text(i + 0.2, df_plot['% it'].iloc[i] + 0.5, f"{df_plot['% it'].iloc[i]:.2f}", color='black', ha='center', fontsize=14)
    
    # Add labels and title
    plt.title('')
    plt.ylabel('Pourcentage', fontsize=14)
    plt.xlabel('')
    plt.legend(fontsize=14)
    plt.xticks(rotation=45, fontsize=14)  # <-- This line rotates the x-axis labels
    plt.yticks(fontsize=14)
    
    max_value = df_plot[['% fr', '% it']].max().max()  # Get the maximum value from both columns
    plt.ylim(0, max_value + 2)  # Increase upper limit of y-axis
    
    # Show the plot
    plt.tight_layout()  # Ensure everything fits within the figure area
    
    # Save as PDF or SVG for vector quality (no resolution loss)
    #↕plt.savefig('high_quality_plot.pdf', bbox_inches='tight')  # Save as vector PDF (best quality)
    # Alternatively, you could save as SVG:
    # plt.savefig('high_quality_plot.svg', bbox_inches='tight')
    
    # If you prefer raster image format (PNG), use very high DPI:
    plt.savefig(project_dir/"plots/typologie_français_italien.png", dpi=1200, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    #############filter out types in chain start and types in chain 2nd position

#solo CATENE
plot1 = plot_barres(french, italian)

##check french content
print("\nsum french : ",french['Nb'].sum())

#plot su anaforici, distribuzione per posizione uno e due ?

plot2 = plot_barres(french_1, italian_1)
plot3 = plot_barres(french_2, italian_2)    
    
def plot_barres_horizontal(french, italian, name):
    # Fusionner et calculer les pourcentages
    
    #fixed categories for x axis label order
    categories = ["Pronom", "SN défini", "SN indéfini","SN possessif", "SN démonstratif", "SN sans déterminant",
                  "Nom Propre", "Anaphore Zéro", "Autre"]
    
    merged_df = pd.merge(french, italian, on='Type', suffixes=('_fr', '_it'))
    merged_df['% fr'] = merged_df['Nb_fr'] / merged_df['Nb_fr'].sum() * 100
    merged_df['% it'] = merged_df['Nb_it'] / merged_df['Nb_it'].sum() * 100

    sum_row = merged_df[['Nb_fr', 'Nb_it']].sum()
    sum_row['Type'] = 'Total'
    merged_df = merged_df.append(sum_row, ignore_index=True)

    # Couleurs pastel
    pastel_colors = sns.color_palette("pastel")[3:5]

    df_plot = merged_df[merged_df['Type'] != 'Total']
    df_plot['Type'] = pd.Categorical(df_plot['Type'], categories=categories, ordered=True)
    
    
    df_plot = df_plot.sort_values('% fr', ascending=True)

    # Création du graphique en barres horizontales
    ax = df_plot.plot(x='Type', y=['% fr', '% it'], kind='barh', figsize=(12, 8),
                      color=pastel_colors, edgecolor='black')

    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Ajouter les étiquettes de pourcentage à droite des barres
    for i in range(len(df_plot)):
        ax.text(df_plot['% it'].iloc[i] + 0.5, i + 0.2, f"{df_plot['% it'].iloc[i]:.2f}", va='center', fontsize=14)
        ax.text(df_plot['% fr'].iloc[i] + 0.5, i - 0.2, f"{df_plot['% fr'].iloc[i]:.2f}", va='center', fontsize=14)

    plt.xlabel('Pourcentage', fontsize=16)
    plt.ylabel('')
    plt.legend(fontsize=16, loc='lower right')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    max_value = df_plot[['% fr', '% it']].max().max()
    plt.xlim(0, max_value + 5)

    plt.tight_layout()
    plt.savefig(name, dpi=1200, bbox_inches='tight')
    plt.show()

#toutes mentions confondues 
plot_barres_horizontal(french, italian, project_dir/"plots/typologie_français_italien_horizontal.png")





plot_barres_horizontal(french_1, italian_1, project_dir/"plots/typologie_français_italien_CE1.png")
plot_barres_horizontal(french_2, italian_2, project_dir/"plots/typologie_français_italien_CE2.png")


# --- Funzione per costruire la matrice posizione × tipo (rimane uguale) ---
def build_position_type_matrix(dfs_dict, sheet_groups, max_position=5):
    records = []

    for sheet_name, df in dfs_dict.items():
        if 'tag_occurrences' in df.columns:
            df = df.dropna(subset=['tag_occurrences'])  
            df['tag_occurrences'] = pd.to_numeric(df['tag_occurrences'], errors='coerce')
            df = df[df['tag_occurrences'] <= max_position]  

            group_type = sheet_groups.get(sheet_name, 'Autre')
            counts = df['tag_occurrences'].value_counts()
            for pos, count in counts.items():
                records.append({'Position': int(pos), 'Type': group_type, 'Count': count})

    matrix_df = pd.DataFrame(records)
    pivot = matrix_df.pivot_table(index='Position', columns='Type', values='Count', aggfunc='sum', fill_value=0)

    categories = ["Pronom", "SN défini", "SN indéfini", "SN possessif", "SN démonstratif",
                  "SN sans déterminant", "Nom Propre", "Anaphore Zéro", "Autre"]
    pivot = pivot.reindex(columns=categories, fill_value=0)

    return pivot.sort_index()

# --- Funzione modificata per plottare la heatmap su un asse specifico ---
def plot_heatmap(matrix_df, ax, language='Français'):
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, cbar=False, ax=ax,
               annot_kws={"size": 20})
    ax.set_title(f"Distribution des types de mentions par position ({language})", fontsize=22)
    ax.set_ylabel("Position dans la chaîne", fontsize=24)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=22)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=22)

# --- Creazione matrici ---
matrix_fr = build_position_type_matrix(filtered_fr, sheet_groups, max_position=5)
matrix_it = build_position_type_matrix(filtered_ita, sheet_groups, max_position=5)

# --- Plot in subplot orizzontale ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

plot_heatmap(matrix_fr, axes[0], language='Français')
plot_heatmap(matrix_it, axes[1], language='Italien')

plt.savefig(project_dir/"plots/heatmap.png")
 
    
plt.tight_layout()
plt.show()
