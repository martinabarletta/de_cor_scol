# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:40:23 2025

@author: matil

check doubles
"""

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
    
    # Display the DataFrame
    print(sheet_info_df)
    
    # Convert to DataFrame
    sheet_info_df = pd.DataFrame(sheet_info)
    
    # Now, group by 'Group' and sum the 'Row Count'
    grouped_df = sheet_info_df.groupby("Type")["Nb"].sum().reset_index()
    
    # Display the grouped DataFrame
    print(grouped_df)
    return grouped_df, dfs




# Load the Excel file
file_path_fr = "typologie_mentions_fr_V3_DEF.xlsx"
file_path_it = "ita_types_v2.xlsx"

#toutes mentions
french, fr_dfs = calculate_ERType(file_path_fr, sheet_groups)
italian, it_dfs = calculate_ERType(file_path_it, sheet_groups)

#####filter out singletons and anaphoras##############################################################################
filtered_df_fr = pd.read_csv('./anaphores_fr.csv') #<- fatto con interdistance_table
filtered_df_it = pd.read_csv('./anaphores_it.csv') #<- fatto con interdistance_table


def filter_anaphoras(dfs_dict, filtered_df) : #dico de df, df anaphores
    
    # Define the columns to compare
    comparison_columns = ['begin', 'end', 'mention', 'tag', 'tag_occurrences', 'Source']
    
    # Ensure filtered_df only contains relevant columns
    filtered_df_selected = filtered_df[comparison_columns].copy()
    
    # Convert data types in filtered_df_selected
    for col in comparison_columns:
        if col in filtered_df_selected.columns:
            if filtered_df_selected[col].dtype == 'object':
                filtered_df_selected[col] = filtered_df_selected[col].astype(str).str.strip()
            else:
                filtered_df_selected[col] = pd.to_numeric(filtered_df_selected[col], errors='coerce')
    
    # Iterate over each DataFrame in the dictionary and drop matching rows
    for key in dfs_dict:
        existing_columns = [col for col in comparison_columns if col in dfs_dict[key].columns]
        
        # Convert data types in dfs_dict[key] to match filtered_df_selected
        for col in existing_columns:
            if dfs_dict[key][col].dtype == 'object':
                dfs_dict[key][col] = dfs_dict[key][col].astype(str).str.strip()
            else:
                dfs_dict[key][col] = pd.to_numeric(dfs_dict[key][col], errors='coerce')
    
        # Merge and filter
        dfs_dict[key] = dfs_dict[key].merge(filtered_df_selected[existing_columns], on=existing_columns, how='left', indicator=True)
        dfs_dict[key] = dfs_dict[key][dfs_dict[key]['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    return dfs_dict
######

filtered_fr = filter_anaphoras(fr_dfs, filtered_df_fr)
filtered_ita = filter_anaphoras(it_dfs, filtered_df_it)

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
    
    # Display the DataFrame
    print(sheet_info_df)
    
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
    categories = ["Pronom", "SN défini", "SN indéfini","SN possessif", "SN démonstratif", "SN sans déterminant",
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
    plt.savefig('./typologie_français_italien.png', dpi=1200, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    #############filter out types in chain start and types in chain 2nd position
    
plot1 = plot_barres(french, italian)
plot2 = plot_barres(french_1, italian_1)
plot3 = plot_barres(french_2, italian_2)    
    
def plot_barres_horizontal(french, italian):
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
    df_plot = df_plot.sort_values('Type', ascending=True)[::-1]


    # Création du graphique en barres horizontales
    ax = df_plot.plot(x='Type', y=['% fr', '% it'], kind='barh', figsize=(12, 8),
                      color=pastel_colors, edgecolor='black')

    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Ajouter les étiquettes de pourcentage à droite des barres
    for i in range(len(df_plot)):
        ax.text(df_plot['% fr'].iloc[i] + 0.5, i - 0.2, f"{df_plot['% fr'].iloc[i]:.2f}", va='center', fontsize=14)
        ax.text(df_plot['% it'].iloc[i] + 0.5, i + 0.2, f"{df_plot['% it'].iloc[i]:.2f}", va='center', fontsize=14)

    plt.xlabel('Pourcentage', fontsize=14)
    plt.ylabel('')
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    max_value = df_plot[['% fr', '% it']].max().max()
    plt.xlim(0, max_value + 2)

    plt.tight_layout()
    plt.savefig('./typologie_français_italien_horizontal.png', dpi=1200, bbox_inches='tight')
    plt.show()


plot_barres_horizontal(french, italian)
plot_barres_horizontal(french_1, italian_1)
plot_barres_horizontal(french_2, italian_2)
