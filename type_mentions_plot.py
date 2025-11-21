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
import numpy as np

# -------------------------
# Config
# -------------------------
sheet_groups = {
    "sn_def": "SN défini",
    "sn_indef": "SN indéfini",
    "verb": "Anaphore Zéro",
    "pron": "Pronom",
    "propn": "Nom Propre",
    "det_poss": "SN possessif",
    "sn_poss": "SN possessif",
    "sn_dem": "SN démonstratif",
    "sn_no_det": "SN sans déterminant",
    "numerals": "Autre",
    "autre": "Autre"
}

categories_order = ["Pronom", "SN défini", "SN indéfini",
                    "SN possessif", "SN démonstratif", "SN sans déterminant",
                    "Nom Propre", "Anaphore Zéro", "Autre"]

# Paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sheets_dir = project_dir / "sheets"
plots_dir = project_dir / "plots"
sheets_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# Input files
file_path_fr = project_dir / "sheets" / "francais" / "typologie_mentions_fr_V4.xlsx"
file_path_it = project_dir / "sheets" / "italien" / "ita_types_v2.xlsx"

# CSVs filtered
filtered_csv_fr = project_dir / "sheets" / "francais" / "anaphores_fr.csv"
filtered_csv_it = project_dir / "sheets" / "italien" / "anaphores_it.csv"

# -------------------------
# Utilities
# -------------------------
def add_school_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str)
        df["School_Level"] = df["Source"].apply(
            lambda s: "CE1" if "CE1" in s else ("CE2" if "CE2" in s else pd.NA)
        )
    else:
        df["School_Level"] = pd.NA
    return df

def read_excel_sheets_with_index(file_path: Path) -> dict:
    xls = pd.ExcelFile(file_path)
    dfs = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if "Unnamed: 0" in df.columns:
            dfs[sheet] = df.dropna(how="all").reset_index(drop=True)
    return dfs

def calculate_ERType(file_path: Path, sheet_groups: dict):
    dfs = read_excel_sheets_with_index(file_path)
    sheet_info = []
    for sheet_name, df in dfs.items():
        row_count = len(df)
        group_name = sheet_groups.get(sheet_name, "Autre")
        sheet_info.append({"Sheet Name": sheet_name, "Type": group_name, "Nb": row_count})
    sheet_info_df = pd.DataFrame(sheet_info)
    grouped_df = sheet_info_df.groupby("Type")["Nb"].sum().reset_index()
    print(f"Grouped counts for {file_path.name}:\n", grouped_df)
    return grouped_df, dfs

def clean_and_prepare_filtered(filtered_df: pd.DataFrame, comparison_columns=None) -> pd.DataFrame:
    if comparison_columns is None:
        comparison_columns = ['begin', 'end', 'mention', 'tag', 'tag_occurrences', 'Source']
    cols = [c for c in comparison_columns if c in filtered_df.columns]
    df = filtered_df[cols].copy()
    for c in cols:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def filter_anaphoras(dfs_dict: dict, filtered_df: pd.DataFrame, comparison_columns=None):
    if comparison_columns is None:
        comparison_columns = ['begin', 'end', 'mention', 'tag', 'tag_occurrences', 'Source']
    filtered_df_selected = clean_and_prepare_filtered(filtered_df, comparison_columns)
    dropped_rows_list = []
    filtered_out_dfs = {}
    for key, df in dfs_dict.items():
        df_work = df.copy()
        existing_columns = [c for c in comparison_columns if c in df_work.columns]
        for col in existing_columns:
            if df_work[col].dtype == "object":
                df_work[col] = df_work[col].astype(str).str.strip()
            else:
                df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        merged = df_work.merge(filtered_df_selected[existing_columns],
                               on=existing_columns, how="left", indicator=True)
        dropped = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        if not dropped.empty:
            dropped["origin"] = key
            dropped_rows_list.append(dropped)
        kept = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        filtered_out_dfs[key] = kept.reset_index(drop=True)
    if dropped_rows_list:
        dropped_df = pd.concat(dropped_rows_list, ignore_index=True)
    else:
        dropped_df = pd.DataFrame(columns=(comparison_columns + ["origin"]))
    return filtered_out_dfs, dropped_df

def separate_anaphoras_and_singletons(dropped_df: pd.DataFrame):
    if dropped_df.empty:
        return pd.DataFrame(), pd.DataFrame(), dropped_df.copy()

    df = dropped_df.copy()
    # normalize tag_occurrences
    df["tag_occurrences"] = pd.to_numeric(df.get("tag_occurrences", pd.NA), errors="coerce")
    # group by Source + tag
    grouped = df.groupby(["Source", "tag"], dropna=False)

    anaphoras_parts = []
    singletons_parts = []
    other_parts = []

    for (source, tag), group in grouped:
        occ = group["tag_occurrences"].dropna().astype(int).tolist()
        # Conditions robusti per identificare coppie 1 & 2
        has1 = 1 in occ
        has2 = 2 in occ
        # count how many valid occurrences 1 or 2
        count_1 = occ.count(1)
        count_2 = occ.count(2)

        if has1 and has2 and count_1 >= 1 and count_2 >= 1:
            # consider it an anaphora: keep the rows that correspond to occ 1/2
            anaphora_rows = group[group["tag_occurrences"].isin([1,2])]
            anaphoras_parts.append(anaphora_rows)
        elif has1 and not has2 and count_1 >= 1 and len(group) == count_1:
            # singleton (only occurrence 1 present)
            singletons_parts.append(group)
        else:
            other_parts.append(group)

    anaphoras_df = pd.concat(anaphoras_parts, ignore_index=True) if anaphoras_parts else pd.DataFrame()
    singletons_df = pd.concat(singletons_parts, ignore_index=True) if singletons_parts else pd.DataFrame()
    # other_parts contiene righe ambigue / anomale
    other_df = pd.concat(other_parts, ignore_index=True) if other_parts else pd.DataFrame()

    return anaphoras_df, singletons_df, df.copy()

def map_category(df: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    df = df.copy()
    df[origin_col] = df.get(origin_col, "").astype(str)
    df["Category_mapped"] = df[origin_col].map(sheet_groups).fillna("Autre")
    return df


def concat_dfs_with_origin(dfs_dict: dict) -> pd.DataFrame:
    rows = []
    for origin, df in dfs_dict.items():
        tmp = df.copy()
        tmp["origin"] = origin
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



def pivot_compare_table(counts_df: pd.DataFrame, left_label: str) -> pd.DataFrame:
    if counts_df.empty:
        return pd.DataFrame()
    pivot = counts_df.pivot_table(
        index="Category_mapped",
        columns="School_Level",
        values="Count",
        aggfunc="sum",
        fill_value=0
    )
    pivot = pivot[[col for col in pivot.columns if col in ['CE1', 'CE2']]]
    pivot.columns = [f"{left_label}_{col}" for col in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"Category_mapped": "Category"})
    return pivot


def counts_for_chains(dfs_dict: dict) -> pd.DataFrame:
    df_all = concat_dfs_with_origin(dfs_dict)
    if df_all.empty:
        return pd.DataFrame(columns=["Category_mapped", "School_Level", "Count"])
    df_all = add_school_level(df_all)
    df_all = map_category(df_all, origin_col="origin")
    df_all = df_all[df_all['School_Level'].notna()]
    grouped = df_all.groupby(["Category_mapped", "School_Level"]).size().reset_index(name="Count")
    return grouped


def counts_for_singletons(singletons_df: pd.DataFrame) -> pd.DataFrame:
    df = add_school_level(singletons_df)
    df = map_category(df, origin_col="origin")
    df = df[df['School_Level'].notna()]
    grouped = df.groupby(["Category_mapped", "School_Level"]).size().reset_index(name="Count")
    return grouped



def counts_for_anaphoras(anaphoras_df: pd.DataFrame) -> pd.DataFrame:
    df = add_school_level(anaphoras_df)
    df = map_category(df, origin_col="origin")
    df = df[df['School_Level'].notna()]
    # Conta righe per ogni (Source,tag,Category,School_Level)
    grouped = df.groupby(["Source", "tag", "Category_mapped", "School_Level"], dropna=False).size().reset_index(name="n_rows")
    # Somma n_rows per Category e School_Level (questa è la correzione importante)
    result = grouped.groupby(["Category_mapped", "School_Level"], dropna=False)["n_rows"].sum().reset_index(name="Count")
    return result




def build_comparative_table(chains_counts_fr: pd.DataFrame,
                            chains_counts_it: pd.DataFrame,
                            label_fr: str = "FR",
                            label_it: str = "IT") -> pd.DataFrame:
    left_pivot = pivot_compare_table(chains_counts_fr, label_fr)
    right_pivot = pivot_compare_table(chains_counts_it, label_it)
    merged = pd.merge(left_pivot, right_pivot, on="Category", how="outer").fillna(0)
    expected_cols = ["Category",
                     f"{label_fr}_CE1", f"{label_fr}_CE2",
                     f"{label_it}_CE1", f"{label_it}_CE2"]
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = 0
    merged = merged[expected_cols]
    merged["Total"] = merged[[c for c in merged.columns if c != "Category"]].sum(axis=1)
    merged = merged.sort_values("Total", ascending=False).drop(columns=["Total"]).reset_index(drop=True)
    return merged

# -------------------------
# Plot functions
# -------------------------
def plot_bar_single_language(df, language=None, save_path=None):
    df_plot = df.copy()
    df_plot = df_plot[df_plot['Category'].isin(categories_order)]
    df_plot['Category'] = pd.Categorical(df_plot['Category'], categories=categories_order, ordered=True)
    df_plot['%'] = df_plot['Nb'] / df_plot['Nb'].sum() * 100
    df_plot = df_plot.sort_values('Category')

    ax = df_plot.plot(x='Category', y='%',
                      kind='bar', width=0.8, figsize=(12, 8),
                      color=sns.color_palette("pastel")[3], edgecolor='black')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(df_plot['%']):
        ax.text(i, val + 0.5, f"{val:.2f}", ha='center', fontsize=14)

    plt.ylabel('Pourcentage', fontsize=14)
    plt.xlabel('')
    if language:
        plt.title(language, fontsize=16)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def prepare_df_for_single_language_plot(df_table, language_prefix):
    level_cols = [col for col in df_table.columns if col.startswith(language_prefix)]
    df_long = df_table[['Category'] + level_cols].copy()
    df_long = df_long.melt(id_vars='Category', value_vars=level_cols,
                           var_name='Level', value_name='Nb')
    df_long['Level'] = df_long['Level'].str.replace(f"{language_prefix}_", "")
    df_long = df_long[df_long['Level'].isin(['CE1', 'CE2'])]  # rimuovo Unknown
    return df_long


def plot_bar_horizontal_with_levels(df_long, language=None, save_path=None):
    """
    Plot barre orizzontali con livelli CE1 e CE2.
    Le percentuali sono calcolate sul totale di tutte le categorie e livelli insieme.
    """
    df_plot = df_long.copy()
    df_plot = df_plot[df_plot['Category'].isin(categories_order)]
    df_plot['Category'] = pd.Categorical(df_plot['Category'], categories=categories_order, ordered=True)
    df_plot = df_plot.sort_values('Category')

    # Pivot per avere le colonne CE1 e CE2
    df_pivot = df_plot.pivot(index='Category', columns='Level', values='Nb').fillna(0)

    # Percentuale sul totale di tutti i valori (somma di CE1 e CE2 di tutte le categorie)
    total = df_pivot.sum().sum()
    df_pivot_percent = df_pivot / total * 100

    # Plot
    bar_width = 0.4
    y = np.arange(len(df_pivot_percent))
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("pastel")[0:2]

    for i, col in enumerate(df_pivot_percent.columns):
        ax.barh(y + i*bar_width, df_pivot_percent[col], height=bar_width,
                label=col, color=colors[i], edgecolor='black')
        for j, val in enumerate(df_pivot_percent[col]):
            if val > 0:
                ax.text(val + 0.5, y[j] + i*bar_width, f"{val:.1f}", va='center', fontsize=12)

    ax.set_yticks(y + bar_width / 2)
    ax.set_yticklabels(df_pivot_percent.index)
    ax.set_xlabel('Pourcentage', fontsize=16)
    
    ax.set_xlim(0, 30)

    if language:
        ax.set_title(language, fontsize=16)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.legend(title='Level')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_bar_horizontal_with_levels_counts(df_long, language=None, save_path=None):
    """
    Plot barre orizzontali con livelli CE1 e CE2.
    I valori plottati sono i conteggi reali (Nb).
    """
    df_plot = df_long.copy()
    df_plot = df_plot[df_plot['Category'].isin(categories_order)]
    df_plot['Category'] = pd.Categorical(df_plot['Category'], categories=categories_order, ordered=True)
    df_plot = df_plot.sort_values('Category')

    # Pivot per avere le colonne CE1 e CE2 con valori reali
    df_pivot = df_plot.pivot(index='Category', columns='Level', values='Nb').fillna(0)

    # Plot
    bar_width = 0.4
    y = np.arange(len(df_pivot))
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("pastel")[4:6]

    for i, col in enumerate(df_pivot.columns):
        ax.barh(
            y + i * bar_width,
            df_pivot[col],
            height=bar_width,
            label=col,
            color=colors[i],
            edgecolor='black'
        )
        # Etichette con valori reali
        for j, val in enumerate(df_pivot[col]):
            if val > 0:
                ax.text(val + 0.5, y[j] + i * bar_width, f"{val}", va='center', fontsize=12)

    ax.set_yticks(y + bar_width / 2)
    ax.set_yticklabels(df_pivot.index)
    ax.set_xlabel('Nombre', fontsize=16)

    if language:
        ax.set_title(language, fontsize=16)

    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.legend(title='Level')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# -------------------------
# Heatmap posizioni
# -------------------------
# def build_position_type_matrix(dfs_dict, max_position=5):
#     records = []
#     for df in dfs_dict.values():
#         if 'tag_occurrences' in df.columns:
#             df = df.dropna(subset=['tag_occurrences'])
#             df['tag_occurrences'] = pd.to_numeric(df['tag_occurrences'], errors='coerce')
#             df = df[df['tag_occurrences'] <= max_position]
#             for sheet, group_type in sheet_groups.items():
#                 if group_type in df.columns:
#                     counts = df[group_type].value_counts()
#                     for pos, count in counts.items():
#                         records.append({'Position': int(pos), 'Type': group_type, 'Count': count})
#     matrix_df = pd.DataFrame(records)
#     if matrix_df.empty:
#         return pd.DataFrame(0, index=range(1,max_position+1), columns=categories_order)
#     pivot = matrix_df.pivot_table(index='Position', columns='Type', values='Count', aggfunc='sum', fill_value=0)
#     pivot = pivot.reindex(columns=categories_order, fill_value=0)
#     return pivot.sort_index()

# def plot_heatmap(matrix_df, ax, language='Français'):
#     sns.heatmap(matrix_df, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, cbar=False, ax=ax,
#                 annot_kws={"size": 16})
#     ax.set_title(f"Distribution des types de mentions par position ({language})", fontsize=18)
#     ax.set_ylabel("Position dans la chaîne", fontsize=16)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

# -------------------------
# Wrapper per tutti i livelli e tipi
# -------------------------
def plot_all_levels_single_language(df_table, name_prefix, language_col_prefix, title_prefix=None):
    df_long = prepare_df_for_single_language_plot(df_table, language_col_prefix)

    # verticale
    plot_bar_single_language(df_long.groupby('Category')['Nb'].sum().reset_index(),
                             language=f"{title_prefix or language_col_prefix}",
                             save_path=plots_dir / f"{name_prefix}_{language_col_prefix}.png")

    # orizzontale
    plot_bar_horizontal_with_levels(df_long,
                                    language=f"{title_prefix or language_col_prefix}",
                                    save_path=plots_dir / f"{name_prefix}_{language_col_prefix}_horizontal.png")


        # orizzontale
    plot_bar_horizontal_with_levels_counts(df_long,
                                    language=f"{title_prefix or language_col_prefix}",
                                    save_path=plots_dir / f"{name_prefix}_{language_col_prefix}_counts.png")


def make_all_plots_single_language(filtered_fr, filtered_it, table_chains, table_anaphoras, table_singletons):
    # Verticale/Orizzontale
    plot_all_levels_single_language(table_chains, "chains", "FR", title_prefix="Distribution des mentions dans les chaines - français")
    plot_all_levels_single_language(table_chains, "chains", "IT", title_prefix="Distribution des mentions dans les chaines - italien")
    plot_all_levels_single_language(table_anaphoras, "anaphoras", "FR", title_prefix="Distribution des mentions dans les anaphores - français")
    plot_all_levels_single_language(table_anaphoras, "anaphoras", "IT", title_prefix="Distribution des mentions dans les anaphores - italien")
    plot_all_levels_single_language(table_singletons, "singletons", "FR", title_prefix="Distribution des singletons - français")
    plot_all_levels_single_language(table_singletons, "singletons", "IT", title_prefix="Distribution des singletons - italien")

    # # Heatmap
    # matrix_fr = build_position_type_matrix(filtered_fr)
    # matrix_it = build_position_type_matrix(filtered_it)
    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # plot_heatmap(matrix_fr, axes[0], language='Français')
    # plot_heatmap(matrix_it, axes[1], language='Italien')
    # plt.tight_layout()
    # heatmap_path = plots_dir / "heatmap_FR_IT.png"
    # plt.savefig(heatmap_path, dpi=1200, bbox_inches='tight')
    # plt.show()
    # print(f"Heatmap salvata in: {heatmap_path}")

# -------------------------
# MAIN
# -------------------------
def main():
    grouped_fr, fr_dfs = calculate_ERType(file_path_fr, sheet_groups)
    grouped_it, it_dfs = calculate_ERType(file_path_it, sheet_groups)
    for k in fr_dfs: fr_dfs[k] = add_school_level(fr_dfs[k])
    for k in it_dfs: it_dfs[k] = add_school_level(it_dfs[k])
    filtered_df_fr = pd.read_csv(filtered_csv_fr)
    filtered_df_it = pd.read_csv(filtered_csv_it)
    filtered_fr, dropped_fr = filter_anaphoras(fr_dfs, filtered_df_fr)
    filtered_it, dropped_it = filter_anaphoras(it_dfs, filtered_df_it)
    dropped_fr = add_school_level(dropped_fr)
    dropped_it = add_school_level(dropped_it)
    anaphoras_fr, singletons_fr, dropped_fr_saved = separate_anaphoras_and_singletons(dropped_fr)
    anaphoras_it, singletons_it, dropped_it_saved = separate_anaphoras_and_singletons(dropped_it)
    
    # Salvo dropped_df su CSV
    dropped_fr_saved.to_csv("./dropped_fr.csv", index=False, encoding="utf-8")
    dropped_it_saved.to_csv("./dropped_it.csv", index=False, encoding="utf-8")
    chains_counts_fr = counts_for_chains(filtered_fr)
    chains_counts_it = counts_for_chains(filtered_it)
    anaphora_counts_fr = counts_for_anaphoras(anaphoras_fr)
    anaphora_counts_it = counts_for_anaphoras(anaphoras_it)
    singleton_counts_fr = counts_for_singletons(singletons_fr)
    singleton_counts_it = counts_for_singletons(singletons_it)
    table_chains = build_comparative_table(chains_counts_fr, chains_counts_it, "FR", "IT")
    table_anaphoras = build_comparative_table(anaphora_counts_fr, anaphora_counts_it, "FR", "IT")
    table_singletons = build_comparative_table(singleton_counts_fr, singleton_counts_it, "FR", "IT")
    table_chains.to_csv(sheets_dir / "comparative_chains_FR_IT.csv", index=False, encoding="utf-8")
    table_anaphoras.to_csv(sheets_dir / "comparative_anaphoras_FR_IT.csv", index=False, encoding="utf-8")
    table_singletons.to_csv(sheets_dir / "comparative_singletons_FR_IT.csv", index=False, encoding="utf-8")
    print("Tabelle comparative salvate.")
    return filtered_fr, filtered_it, table_chains, table_anaphoras, table_singletons

if __name__ == "__main__":
    filtered_fr, filtered_it, table_chains, table_anaphoras, table_singletons = main()
    make_all_plots_single_language(filtered_fr, filtered_it, table_chains, table_anaphoras, table_singletons)


