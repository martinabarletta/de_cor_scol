# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:49:33 2025

@author: matil
"""
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# 1. DATA PROCESSING
# -------------------------------------------------------------------

def count_positive_values(file_path, first_data_col=4):
    """Load a CSV and compute number of positive values across target columns."""
    df = pd.read_csv(file_path)

    # Select columns starting from index `first_data_col`
    target_columns = df.columns[first_data_col:]

    df['nb_referents'] = (df[target_columns] > 0).sum(axis=1)

    # Return only useful columns + computed metric
    result_df = df[[df.columns[1], 'nb_referents']].copy()
    result_df.rename(columns={df.columns[1]: "texte"}, inplace=True)
    return result_df


def enrich_metadata(df, langue):
    """Add language and CE1/CE2 extraction."""
    df["langue"] = langue
    df["niveau"] = df["texte"].str.extract(r"(CE1|CE2)")
    return df


def save_results(df, out_path):
    """Save result CSV."""
    df.to_csv(out_path, index=False)


# -------------------------------------------------------------------
# 2. VISUALIZATION
# -------------------------------------------------------------------

# def plot_violin(df, langue):
#     """Violin plot + annotated stats."""
#     plt.figure(figsize=(7, 5))

#     sns.violinplot(x='niveau', y='nb_referents',
#                    data=df, inner='box', palette='pastel')

#     # Compute stats
#     stats = df.groupby('niveau')['nb_referents'].agg(['mean', 'max', 'min']).reset_index()

#     # Add text annotations
#     for i, row in stats.iterrows():
#         plt.text(i, row['max'], f"Max = {row['max']:.1f}",
#                  ha='center', va='top', fontweight='bold')
#         plt.text(i, row['mean'], f"Moy = {row['mean']:.1f}",
#                  ha='center', va='top', fontstyle='italic')
#         plt.text(i, row['min'], f"Min = {row['min']:.1f}",
#                  ha='center', va='bottom', fontweight='bold')

#     plt.title(f"Distribution du nombre de référents par niveau scolaire – corpus {langue}")
#     plt.xlabel("Niveau scolaire")
#     plt.ylabel("Nombre de référents par texte")
#     plt.tight_layout()
#     plt.show()


def plot_count(df, langue):
    """Barplot of number of texts per number of referents, by level."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='nb_referents',
                       hue='niveau', palette='pastel')

    # Add labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.05,
                int(height),
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

    ax.set_ylim(0, 37)

    plt.title(f"Nombre de textes selon nb de référents – corpus {langue}")
    plt.xlabel("Nombre de référents dans le texte")
    plt.ylabel("Nombre de textes")
    plt.legend(title="Niveau scolaire")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 3. MASTER FUNCTION TO RUN THE FULL PIPELINE
# -------------------------------------------------------------------

def process_language(
        file_path: Path,
        output_path: Path,
        langue: str,
        first_data_col: int = 4
    ):
    """
    Full pipeline: load → compute → enrich → save → plots.
    """
    df = count_positive_values(file_path, first_data_col)
    df = enrich_metadata(df, langue)
    save_results(df, output_path)

    plot_count(df, langue)

    return df

##two barplots next to each other 

def plot_two_barplots(df1, df2, label1, label2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------
    # LEFT BARPLOT
    # ------------------------------
    sns.countplot(
        data=df1,
        x="nb_referents",
        hue="niveau",
        palette="pastel",
        ax=axes[0]
    )
    axes[0].set_title(f"{label1}: Nombre de textes par nombre de référents", fontsize=14)
    axes[0].set_xlabel("Nombre de référents", fontsize=12)
    axes[0].set_ylabel("Nombre de textes", fontsize=12)

    # Add labels
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].text(
                p.get_x() + p.get_width()/2,
                h + 0.1,
                int(h),
                ha='center', va='bottom', fontsize=9
            )

    # ------------------------------
    # RIGHT BARPLOT
    # ------------------------------
    sns.countplot(
        data=df2,
        x="nb_referents",
        hue="niveau",
        palette="pastel",
        ax=axes[1]
    )
    axes[1].set_title(f"{label2}: Nombre de textes par nombre de référents", fontsize=14)
    axes[1].set_xlabel("Nombre de référents", fontsize=12)
    axes[1].set_ylabel("Nombre de textes", fontsize=12)

    # Add labels
    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].text(
                p.get_x() + p.get_width()/2,
                h + 0.1,
                int(h),
                ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 4. EXAMPLE USAGE
# -------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "sheets/italien"

    input_file  = data_dir / "referents_corpus_italien.csv"
    output_file = data_dir / "referents_fr.csv"

    df_it = process_language(
        file_path=input_file,
        output_path=output_file,
        langue="italien"
    )

    data_fr = project_dir / "sheets/francais"
    input_fr  = data_fr / "referents_corpus_francais.csv"
    output_fr = data_fr / "referents_francais_processed.csv"

    df_fr = process_language(
        file_path=input_fr,
        output_path=output_fr,
        langue="français"
    )

    # ------------------------------
    # Combined 2-subplot barplot
    # French (left)    |    Italian (right)
    # ------------------------------
    plot_two_barplots(df_fr, df_it, "Français", "Italien")



