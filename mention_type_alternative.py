# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:37:19 2025

@author: matil
"""

import pandas as pd

#file = "./annotations_francais/toutes_mentions_corpus_detail_interdistance_fr.csv"
file = "./annotations_italien/toutes_mentions_corpus_detail_interdistance_ita.csv"

df = pd.read_csv(file)


# Assuming df is already defined
remaining_df = df.copy()  # Copy the original dataframe to filter progressively

# Extract each category and update remaining_df
sn_def = remaining_df[remaining_df['POSno'].str.contains('NOUN', na=False) & remaining_df['morphNoPunct'].str.startswith("['Definite=Def", na=False)]
remaining_df = remaining_df.drop(sn_def.index)  # Remove classified elements

sn_ind = remaining_df[remaining_df['POSno'].str.contains('NOUN', na=False) & remaining_df['morphNoPunct'].str.startswith("['Definite=Ind", na=False)]
remaining_df = remaining_df.drop(sn_ind.index)

#TODO add criteria
sn_poss = remaining_df[remaining_df['POSno'].str.contains('DET', na=False) & remaining_df['morphNoPunct'].str.contains("Poss=Yes", na=False)
                       & remaining_df['POSno'].str.contains('NOUN', na=False)]
remaining_df = remaining_df.drop(sn_poss.index)

verb = remaining_df[remaining_df['POSno'].str.contains('VERB|AUX', na=False) & ~remaining_df['POSno'].str.contains('NOUN', na=False)]
remaining_df = remaining_df.drop(verb.index)

pron = remaining_df[remaining_df['POSno'] == "['PRON']"]
remaining_df = remaining_df.drop(pron.index)

propn = df[(df['POSno'] == "['PROPN']") | (df['POSno'] == "['PROPN', 'PROPN']")]
remaining_df = remaining_df.drop(propn.index)

det_poss = remaining_df[(remaining_df['POSno'] == "['DET']") & remaining_df['morphNoPunct'].str.contains('Poss=Yes', na=False)]
remaining_df = remaining_df.drop(det_poss.index)

sn_dem = remaining_df[(remaining_df['POSno'].str.startswith("['DET'", na=False)) & remaining_df['morphNoPunct'].str.contains('PronType=Dem', na=False)]
remaining_df = remaining_df.drop(sn_dem.index)

num = remaining_df[(remaining_df['POSno'].str.startswith("['NUM'", na=False))]
remaining_df = remaining_df.drop(num.index)

sn_no_det = remaining_df[remaining_df['POSno'].str.contains('NOUN', na=False) & ~remaining_df['POSno'].str.startswith("['DET", na=False)]
remaining_df = remaining_df.drop(sn_no_det.index)


# Collect remaining elements into "autre"
autre = remaining_df.copy()

# Write to Excel
with pd.ExcelWriter('ita_types.xlsx') as writer:
    sn_def.to_excel(writer, sheet_name='sn_def', index=False)
    sn_ind.to_excel(writer, sheet_name='sn_indef', index=False)
    sn_poss.to_excel(writer, sheet_name='sn_poss', index=False)
    verb.to_excel(writer, sheet_name='verb', index=False)
    pron.to_excel(writer, sheet_name='pron', index=False)
    propn.to_excel(writer, sheet_name='propn', index=False)
    det_poss.to_excel(writer, sheet_name='det_poss', index=False)
    sn_dem.to_excel(writer, sheet_name='sn_dem', index=False)
    sn_no_det.to_excel(writer, sheet_name='sn_no_det', index=False)
    num.to_excel(writer, sheet_name='numerals', index=False)
    autre.to_excel(writer, sheet_name='autre', index=False)  # Write remaining elements
    
   
print("Excel file 'ita_types.xlsx' created successfully!")
