# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:47:54 2025

@author: matil
"""

# -*- coding: utf-8 -*-
"""
ONE FILE VERSION : 
Created on Thu Jan  9 17:26:37 2025

@author: matil

Le brouillon initial de ce code a été réalisé en utilisant ChatGPT, puis il a été modifié
selon les nécessités de mon travail de thèse.

dernière version des stats pour annotations INCEpTION

- nb tokens (avec, sans ponctuation)
- nb de mentions avec étiquettes (mention chevauchées comptées une fois)
- distance entre mentions


input - merged folders (francais, italien)
output - sheets folders (francais, italien)

"""
import os
import pandas as pd
from cassis import *
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


###############################################################################
##load TypeSystem - contains the annotation layers used in the code - tjrs le meme
with open('./merged/italien/TypeSystem.xml', 'rb') as f:
    typesystem = load_typesystem(f)

#CONSTANTS name of annotation layers that we use further - always the same
token_type_name = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'
mention_type_name = 'webanno.custom.Mentions'
morph='de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures'


token_type = typesystem.get_type(token_type_name)
mention_type = typesystem.get_type(mention_type_name)
morph_type = typesystem.get_type(morph)
###############################################################################

def load_cas_xmi(file_path):
    """
    Load a CAS XMI file using cassis
    Args:
        file_path - XMI file path
    Returns:
        cas - CAS object containing the annotations
    """
    with open(file_path, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)  
        #lenient=True - Leniency helps bypass MetaData errors if any
    return cas

###############################################################################
# NB TOKENS
#count nb of tokens for text (1) and nb tokens without ponctuation for text (2)

def count_tokens(cas):
    """
    

    Parameters
    ----------
    cas : object cas containing annotations

    Returns
    -------
    nb_tokens : nb d'objets dans le niveau Token de l'objet CAS

    """
    tokens = list(cas.select(token_type))
    nb_tokens = len(tokens)
    print(f"longueur du texte nb tokens : {nb_tokens}")
    return nb_tokens

###############################################################################

# (2) nb tokens without punctuation : filter tokens by POS tag
def count_tokens_nopunct(cas):
    tokens = list(cas.select(token_type))
    non_punct_tokens = [token for token in tokens if getattr(token, 'pos', None).coarseValue != 'PUNCT']
    nb_tokens_nopunct = len(non_punct_tokens)
    print(f"longueur du texte nb tokens sans PUNCT : {nb_tokens_nopunct}")    
    return nb_tokens_nopunct

# def count_tokens_nouns(cas):
#     """ NOT USED count nb of NOUNS in tokens
#     """
#     tokens = list(cas.select(token_type))
#     non_punct_tokens = [token for token in tokens if getattr(token, 'pos', None).coarseValue == 'NOUN']
#     nb_tokens_nopunct = len(non_punct_tokens)
#     print(f"Tokens qui sont des NOUNS : {nb_tokens_nopunct}")    
#     return nb_tokens_nopunct

###############################################################################

def extract_mention_details(cas, token_type, mention_type):
    """
    Extracts details of mentions, including: mention text, begin index, end index (caracters not tokens), 
    mention tag, nb of tokens in mention, POS of each token in the mention, POS of each token filtering PUNCT tokens

    Arguments:
        cas: The CAS object containing annotations.
        token_type_name: The type name for tokens (e.g., 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token').
        mention_type_name: The type name for mentions (e.g., 'webanno.custom.Mentions').

    Returns:
        A list of lists, where each sub-list contains mention text, begin index, end index, mention tag, 
        and POS of each token in the mention expression.
    """

    mentions = list(cas.select(mention_type))
    tokens = list(cas.select(token_type))
    details = []
    
    # text, begin and end character, mention tag
    for mention in mentions:
        mention_text = mention.get_covered_text()
        begin_index, end_index = mention.begin, mention.end
        mention_tag = mention.mention

        pos_list, morph_list, pos_list_no_punct, morph_list_no_punct = [], [], [], []

        for token in tokens:
            if token.begin >= begin_index and token.end <= end_index:
                pos_annotation = getattr(token, 'pos', None)
                morph_annotation = getattr(token, 'morph', None)

                if pos_annotation:
                    pos_list.append(pos_annotation.coarseValue)
                if morph_annotation:
                    morph_list.append(morph_annotation.value)
                    
        
        # POS of tokens within the mention span no punctuation
        pos_list_no_punct = []
        morph_list_no_punct = []
        for token in tokens:
            if token.begin >= begin_index and token.end <= end_index:
                pos_annotation = getattr(token, 'pos', None)
                morph_annotation = getattr(token, 'morph', None)
                if pos_annotation.PosValue != 'PUNCT' :
                    pos_list_no_punct.append(pos_annotation.coarseValue)
                if morph_annotation:
                    morph_list_no_punct.append(morph_annotation.value)
                        
        
        #TODO add morphological details 

        details.append([mention_text, begin_index, end_index, mention_tag, 
                                pos_list, morph_list, 
                                pos_list_no_punct, morph_list_no_punct])
    
    return details


def create_mentions_df(details):
    """
    Parameters :
    ----------
    details : A list of lists, where each sub-list contains mention text, 
    begin index, end index, mention tag, and POS of each token in the mention expression.

    Returns
    -------
    pandas DataFrame containing mention text, begin and end caracter, mention tag (flattened
    for overlapping mentions), POS list with and without punctuation, lenght of mention in tokens with/
    without punctuation, 

    """
    #transform datas into DataFrame and add mention len --> one DataFrame for each text here
    columns=['mention', 'begin', 'end', 'tag', 'POS', 'morph', 'POSno', 'morphNoPunct']
    df = pd.DataFrame(details, columns=columns)
    
    df['mentionLen'] = df['POS'].apply(len)
    df['mentionLenNoPunct'] = df['POSno'].apply(len)
    
    # Group by 'begin' and 'end' and aggregate the 'tag' column - count only once overlapping mentions
    ment_details = (
        df.groupby(['begin', 'end'], as_index=False)
        .agg({
            'mention': 'first',   # Take the first mention (or customize as needed)
            'tag': list,          # Combine tags into a list
            'POS': 'first',       # Take the first POS (or customize)
            'morph': 'first', 
            'mentionLen': 'first', # Take the first mentionLen (or customize)
            'POSno': 'first',
            'morphNoPunct': 'first', 
            'mentionLenNoPunct' : 'first'
        })
    )
    
    # flatten tag lists - tag for overlapping mentions -> list of tags, one row
    # normalize tags order by sorting alphabetically
    #ment_details['tag'] = ment_details['tag'].apply(lambda tags: list(set(tags)))
    ment_details['tag'] = ment_details['tag'].apply(lambda tags: sorted(list(set(tags))))
    return ment_details

###V2 take into account overlapping mentions in both languages
def calcul_distance(df):
    """
    Parameters
    ----------
    df : DataFrame qui contient 

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    df['distance'] = 0  # Initialize distance column
        
    for i in range(1, len(df)): #début deuxième mention
        curr_begin = df.loc[i, 'tokenBegin']
        curr_end = df.loc[i, 'tokenEnd']
        prev_begin = df.loc[i - 1, 'tokenBegin']
        prev_end = df.loc[i - 1, 'tokenEnd']

        # Case 1: Distant spans (-1 to take into account current token)
        # si début mention courante est après la fin de la mention précedente -> 
        if curr_begin > prev_end:
        #distance rajouté dans ligne de mention précedente
        #c'est la distance entre cette mention et la suivante 
            df.loc[i, 'distance'] = curr_begin - prev_end - 1

        ## Case 2: Mentions imbriquées
        ###si le début est le meme, la distance depuis ment précédente est de 0
        ## ((son) chat)
        if curr_begin == prev_begin and curr_end > prev_end :
            df.loc[i, 'distance'] = 0
        
        # # Case 3: Smaller span inside larger span 
        # (ita) il [loro] amico
        # come calcolare ?
        elif curr_begin > prev_begin and curr_end < prev_end:
            df.loc[i, 'distance'] = 0  # indiquer avec val diff ?


        # # Case 4: Same ending, smaller span
        # elif curr_end == prev_end and curr_begin > prev_begin:
        #     df.loc[i, 'distance'] = -3  # Arbitrary value for identification


        #distance de la première mention depuis le début du texte
        df.loc[0, 'distance'] = df.loc[0, 'tokenBegin']

    return df


def determine_begin_token(cas, ment_details) :
    """
    contient calcul distance pour rajouter au df la colonne de distance entre mentions
    le valeur indique la distance entre la fin de la mention précédente et le début 
    de la mention suivante - première mention 0 ? dernière mention ?
    
    permet de trouver le token de début et de fin des mentions puis
    utilise calcul_distance pour rajouter col distance au df 

    Parameters    
    cas : TYPE
    ment_details : pandas DataFrame avec détails des mentions 
    (indice début et fin, étiquette etc.)

    Returns
    -------
    ment_details : pandas DataFrame en entrée avec colonne des distances en plus
        DESCRIPTION.

    """
    ##compter distance entre tokens
    tokens = list(cas.select(token_type))
    #mentions = list(cas.select(mention_type))
    
    #filter out punctuation tokens from the text
    non_punct_tokens = [token for token in tokens if getattr(token, 'pos', None).coarseValue != 'PUNCT']
    
    #à partir des mentions, on calcule le token de début de chaque mention
    #listes vides pour indices de début et de fin de la mention    
    mention_indices_begin = []
    mention_indices_end = []
    
    # Match tokens with mentions and track indices
    for mention in ment_details.itertuples():  # Iterate over rows in ment_details df
        begin_index = None
        end_index = None
        #iterate over tokens in text
        for i, token in enumerate(non_punct_tokens):
            if token.begin == mention.begin and begin_index is None:
                begin_index = i
            if token.end == mention.end and end_index is None:
                end_index = i
        
        mention_indices_begin.append(begin_index)
        mention_indices_end.append(end_index)
        
    if len(mention_indices_begin) != len(ment_details):
        raise ValueError("Length mismatch: 'mention_indices_begin' and 'ment_details' row count do not match.")

    if len(mention_indices_end) != len(ment_details):
        raise ValueError("Length mismatch: 'mention_indices_end' and 'ment_details' row count do not match.")
    
    ment_details['tokenBegin'] = mention_indices_begin
    ment_details['tokenEnd'] = mention_indices_end
    
    ment_details = calcul_distance(ment_details)
    
    return ment_details

def position_in_chain(df, tag_column="tag"):
    
    """
    associer index à chaque mention correspondant à sa position dans sa chaine
    ex. deuxième mention de la chaine "cat" = 2
    """
    tag_counts = {}
    occurrences = []
    
    for tags in df[tag_column]:
        tag_tuple = tuple(tags) if isinstance(tags, list) else (tags,)
        
        if tag_tuple not in tag_counts:
            tag_counts[tag_tuple] = 1
        else:
            tag_counts[tag_tuple] += 1
        
    occurrences.append(tag_counts[tag_tuple])

    df["tag_occurrences"] = occurrences
    
    return df

def mentions_by_tag(cas, mention_type_name):
    """
    Regroupe les mentions par étiquette. 
    Overlapping mentions are counted once for each  and the two or more tags put together
    TODO add overlapping mentions option in dictionary ? how ?

    Parameters
    ----------
    cas : TYPE
        DESCRIPTION.
    mention_type_name : TYPE
        DESCRIPTION.

    Returns
    -------
    ment_by_tag : TYPE
        DESCRIPTION.

    """
    mention_type = typesystem.get_type(mention_type_name)
    mentions = list(cas.select(mention_type))
    
    # Organize mentions by tag
    ment_by_tag = {}
    for mention in mentions:
        tag = mention.mention
        if tag not in ment_by_tag:
            ment_by_tag[tag] = []
        ment_by_tag[tag].append(mention)

    return ment_by_tag

def mentions_per_entity(data_dict, key_order):
    """
    Returns the lengths of the value lists in a dictionary, following a specific order of keys.
    
    Parameters:
    - data_dict (dict): The input dictionary where keys map to lists of objects.
    - key_order (list): A list specifying the desired order of keys.

    Returns:
    - list: A list of lengths corresponding to the key order, with 0 or None for missing keys.
    """
    result = []
    for key in key_order:
        if key in data_dict:
            result.append(len(data_dict[key]))
        else:
            result.append(0)  # Change to None if None is preferred
    return result


def add_tag_occurrence_column(df, tag_column="tag"):
    #nuñéroter les mentions meme tag pour ordre
    tag_counts = {}
    occurrences = []

    for tags in df[tag_column]:
        tag_tuple = tuple(sorted(tags)) if isinstance(tags, list) else (tags,)

        if tag_tuple not in tag_counts:
            tag_counts[tag_tuple] = 1
        else:
            tag_counts[tag_tuple] += 1
        
        occurrences.append(tag_counts[tag_tuple])
    
    df["tag_occurrences"] = occurrences
    return df


# Function to convert tags containing commas
def convert_tags(tag_str):
    # Replace commas with '+'
    a = tag_str.replace("[","")
    b = a.replace("]","")
    c = b.replace(" ","")
    d = c.replace("'","")
    return d.replace(',', '+')


#before recomposing apply calcul distance on each group        
# Function to calculate distances
def calcul_interdistance(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    df = df.reset_index(drop=True)  # Reset index to ensure sequential order
    df['interdistance'] = 0  # Initialize distance column

    for i in range(1, len(df)):  # Start from the second row
    
        curr_begin = df.loc[i, 'tokenBegin']
        curr_end = df.loc[i, 'tokenEnd']
        prev_begin = df.loc[i - 1, 'tokenBegin']
        prev_end = df.loc[i - 1, 'tokenEnd']

        # Case 1: Distant spans
        if curr_begin > prev_end:
            df.loc[i, 'interdistance'] = curr_begin - prev_end - 1

        # Case 2: Overlapping mentions (nested)
        elif curr_begin == prev_begin and curr_end > prev_end:
            df.loc[i, 'interdistance'] = 0

        # Case 3: Smaller span inside a larger span
        elif curr_begin > prev_begin and curr_end < prev_end:
            df.loc[i, 'interdistance'] = 0  

    return df

    
def unify_csv_files(folder_path, filter_substring=None):
    """
    Combines all CSV files in the specified folder into a single Pandas DataFrame.
    Uses the header from the first file and skips the first row for subsequent files.
    Adds a column with a substring extracted from the filename.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        filter_substring (str, optional): Substring to filter files by name. 
                                          Only files containing this substring will be included.

    Returns:
        pd.DataFrame: Unified DataFrame containing data from all CSV files.
    """
    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    # Apply optional filter
    if filter_substring:
        csv_files = [f for f in csv_files if filter_substring in f]
    
    if not csv_files:
        print("No files matched the criteria or the folder is empty.")
        return pd.DataFrame()
    
    # Create an empty list to store DataFrames
    dataframes = []
    
    header = None  # To store the header from the first file
    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
                
        try:
            if i == 0:
                # Read the first file normally, and set its header
                df = pd.read_csv(file_path)
                header = df.columns  # Save header from the first file
            else:
                # For subsequent files, skip the first row
                df = pd.read_csv(file_path, skiprows=1, header=None)
                df.columns = header  # Assign header to the current DataFrame
            
            # Add the extracted value as a new column
            df['Source'] = file
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Concatenate all DataFrames
    unified_df = pd.concat(dataframes, ignore_index=True)
    
    # Step 1: Replace '0' in 'distance' with NaN where 'Index' is 0
    unified_df.loc[unified_df['Unnamed: 0'] == 0, 'distance'] = unified_df.loc[unified_df['Unnamed: 0'] == 0, 'distance'].replace(0, np.nan)
    return unified_df
    
    
#tagset pour francais et italien complet
key_order = [
            'cat', 'witch', 'wolf', 'robot', 'ext1', 'ext2', 'ext3', 'ext4', \
            'ext5', 'ext6', 'ext7', 'ext8', 'ext9', 'cat1', 'cat2', 'cat3', \
            'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'robot1', 'robot2', \
            'robot3', 'robot4', 'robot5', 'witch1', 'witch2', 'witch3', \
            'wolf1', 'wolf2', 'wolf3'
            ]

################  MAIN PROGRAM  ###############################################
folder_path = os.path.dirname('./merged/italien/')

out_folder = './sheets/italien/'
output_folder = './sheets/italien/stats_ita/'

all_res=[]
for filename in os.listdir(folder_path):
    results = []
    if filename.endswith('.xmi'):
        print("Elaborating "+filename)
        file_path = os.path.join(folder_path, filename)
        filename = os.path.splitext(filename)[0]
        # Load CAS
        cas = load_cas_xmi(file_path)
        results.append(filename)
        
        #compter nb tokens
        toks = count_tokens(cas)
        results.append(toks)
        toks_no_punct = count_tokens_nopunct(cas)
        results.append(toks_no_punct)
        
        #creer dataframe 1 per texte qui contient tout détail de mentions
        #obtenir un folder avec un fichier csv par texte avec le détail des mentions
        details = extract_mention_details(cas, token_type, mention_type)
        mention_dataframe = create_mentions_df(details)
        mention_dataframe_2 = determine_begin_token(cas, mention_dataframe)
        mention_dataframe_3 = add_tag_occurrence_column(mention_dataframe_2)
        os.makedirs(output_folder , exist_ok=True)
        mention_dataframe_3.to_csv(output_folder+filename+'.csv', ',', encoding='utf-8')
        
        
        # Group by 'tag', apply function, and recompose
        # Apply the function to convert tags in the 'tag' column
        mention_dataframe_3['tag'] = mention_dataframe_3['tag'].astype(str).apply(convert_tags)
        grouped_df = mention_dataframe_3.groupby('tag', sort=False)  # Keep original order
        
        #from tag lists to tag+tag in alphabetical order
        processed_groups = [calcul_interdistance(group) for _, group in grouped_df]
        recomposed_df = pd.concat(processed_groups).reset_index(drop=True)
        os.makedirs(out_folder+'stats_ita_interdistance/', exist_ok=True)
        recomposed_df.to_csv(out_folder+'stats_ita_interdistance/'+filename+'.csv', ',', encoding='utf-8')
        print("done "+filename)
        
    
        #trouver les tags à rajouter au df 
        ment_tags = mentions_by_tag(cas, mention_type_name)    
        output = mentions_per_entity(ment_tags, key_order)
        results = results + output
        prev_header = ['texte', 'nbTok', 'nbTokNoPunct'] + key_order
        header = prev_header
        all_res.append(results)


#dataframe with nbTok (with/noPunct) and mentions per character in text
#TODO add how many mentions are plural and what characters are involved?

#csv unique avec tout le corpus + colonne nom fichier en dernier
# utile seulement pour nb tokens pas pour le reste

res = pd.DataFrame(all_res, columns = header)
#droppa colonne che contengono solo degli zero ok
res = res.replace(0, np.nan).dropna(axis=1,how="all")
res.to_csv(out_folder+"corpus_italien_v2.csv", ",", encoding="utf-8")


#count nb of characters per text
res['Characters'] = (res.iloc[:, 3:] > 0).sum(axis=1)
char_nb = res['Characters'].value_counts()

filtered_CE1 = res.loc[res['texte'].str.contains('CE1', na=False), 'Characters'].value_counts()
print(filtered_CE1)

filtered_CE2 = res.loc[res['texte'].str.contains('CE2', na=False), 'Characters'].value_counts()
print(filtered_CE2)

df = unify_csv_files(output_folder)  # un seul csv depuis folder
print(df)
df.to_csv(out_folder+"toutes_mentions_corpus_detail_ita_v2.csv", sep=",", encoding="utf8")

folder = out_folder+'stats_ita_interdistance/'
df2 = unify_csv_files(folder)  # un seul csv depuis folder
print(df2)
df2.to_csv(out_folder+"toutes_mentions_corpus_detail_interdistance_ita_v2.csv", sep=",", encoding="utf8")


nb_texts = len(res['texte'])
nb_tokens = res['nbTokNoPunct'].sum()

# Supponiamo che il tuo DataFrame sia df

# Count occurrences of each tag per source - equals to len chains
df_pivot = df.groupby(['Source', 'tag']).size().unstack(fill_value=0)

# # Numero di colonne che contengono almeno un valore diverso da zero
nb_characters = (df_pivot != 0).any(axis=0).sum()

#una menzione per riga - len df = numero tot di menzioni annotate
nb_mentions = len(df)

#contare quanti valori superiori a tre, uguali a due o uguali a uno nel df con len chaines
chaines = (df_pivot >= 3).sum().sum()
anaphores = (df_pivot == 2).sum().sum()
singletons = (df_pivot == 1).sum().sum()

df_pivot['all_mentions'] = df_pivot.select_dtypes(include='number').sum(axis=1)

df_pivot = df_pivot.reset_index()
total_sum_ce1 = df_pivot.loc[df_pivot['Source'].str.contains('CE1', na=False), 'all_mentions'].sum()
total_sum_ce2 = df_pivot.loc[df_pivot['Source'].str.contains('CE2', na=False), 'all_mentions'].sum()

df_pivot = df_pivot.reset_index()

##nb chaines par niveau
chains = df_pivot[df_pivot['Source'].str.contains('CE1', na=False)]
sum_chaines_ce1 = chains.applymap(lambda x: x if isinstance(x, int) and x >= 3 else 0)
mean_ce1_3 = sum_chaines_ce1[sum_chaines_ce1 >= 3].mean().mean()


# Group by 'Source' and 'tag' and perform both operations
result = df2.groupby(['Source', 'tag'], as_index=False).agg({
    'distance': 'sum',
    'tag_occurrences': 'max',
})
# Creazione della nuova colonna 'interdistance'
result['interdistance'] = result['distance'] / result['tag_occurrences']
len_moy_chaines = result['tag_occurrences'].mean()



result.to_csv("./interdistance_ita.csv", sep=",", encoding="utf8")


ce1_len_moy = result.loc[result['Source'].str.contains('CE1', na=False), 'tag_occurrences'].mean()
ce2_len_moy = result.loc[result['Source'].str.contains('CE2', na=False), 'tag_occurrences'].mean()


len_max_chaines = result['tag_occurrences'].max()
ce1_len_max = result.loc[result['Source'].str.contains('CE1', na=False), 'tag_occurrences'].max()
ce2_len_max = result.loc[result['Source'].str.contains('CE2', na=False), 'tag_occurrences'].max()

# moy_chaines_texte = 
# ce1_len_moy = result.loc[result['Source'].str.contains('CE1', na=False), 'tag_occurrences'].mean()
# ce2_len_moy = result.loc[result['Source'].str.contains('CE2', na=False), 'tag_occurrences'].mean()


txt = "./sheets/italien/res_ita.txt"
with open(txt, "w") as file:
    file.write(f"Nb of texts: {nb_texts}\n")
    file.write(f"Nb of tokens: {nb_tokens}\n")
    file.write(f"Nb of characters: {nb_characters}\n")
    file.write(f"Nb of mentions: {nb_mentions}\n")
    # file.write(f"Nb of mentions CE1 : {total_sum_ce1}\n")
    # file.write(f"Nb of mentions CE2 : {total_sum_ce2}\n")
    
    file.write(f"Chaines  >= 3: {chaines}\n")
    file.write(f"Chaines CE1  >= 3: {sum_chaines_ce1}\n")
    # file.write(f"Chaines CE2  >= 3: {chains_ce2}\n")

    file.write(f"Anaphores  = 2: {anaphores}\n")
    # file.write(f"Anaphores CE1  = 2: {anaphores_ce1}\n")
    # file.write(f"Anaphores CE2  = 2: {anaphores_ce2}\n")

    file.write(f"Singletons = 1: {singletons}\n")
    # file.write(f"Singletons CE1 = 1: {singletons_ce1}\n")
    # file.write(f"Singletons CE2 = 1: {singletons_ce2}\n")
 
    

    file.write(f"Len moyenne chaines {len_moy_chaines}\n")
    file.write(f"Len moyenne chaines CE1 {ce1_len_moy}\n")
    file.write(f"Len moyenne chaines CE2 {ce2_len_moy}\n")
    file.write(f"Len max chaines {len_max_chaines}\n")
    file.write(f"Len max chaines CE1 {ce1_len_max}\n")
    file.write(f"Len max chaines CE2 {ce2_len_max}\n")
    # file.write(f"Nb moyen chaines par texte {}\n")
    # file.write(f"Nb moyen chaines par texte CE1 {}\n")
    # file.write(f"Nb moyen chaines par texte CE2 {}\n")
    #file.write(f"Tot.: {chaines+anaphores+singletons}\n")
print(f"File {txt} has been created successfully.")