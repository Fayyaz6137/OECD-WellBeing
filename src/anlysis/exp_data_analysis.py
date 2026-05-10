import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from configs.config import PLOTS_DIR, DATA_PROCESSED_CSV_PATH, PALETTE, LINE_COLOURS

df = pd.read_csv(DATA_PROCESSED_CSV_PATH)
numeric_cols = df.select_dtypes(include='number').columns.tolist()


def kda_analysis():
    print("\nKDA Analysis ...\n")

    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=25, color=PALETTE['teal'], alpha=0.6,
                density=True)  # ← theme color # Histogram using your palette
        ax_kde_color = LINE_COLOURS[i % len(LINE_COLOURS)]  # KDE line using line palette (cycle colors)
        data.plot.kde(ax=ax, color=ax_kde_color, linewidth=2)
        ax.set_title(col, fontsize=9, color=PALETTE['navy'])
        ax.set_xlabel('Indicator Value', fontsize=8, color=PALETTE['gray'])
        ax.set_ylabel('Density', fontsize=8, color=PALETTE['gray'])
        ax.tick_params(colors=PALETTE['gray'])

    # Hide empty plots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'Distribution of Numerical Indicators with KDE',
        fontsize=14,
        y=1.01,
        color=PALETTE['navy']
    )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(PLOTS_DIR, '01_distributions.png'),
        dpi=150,
        bbox_inches='tight'
    )

    plt.show()
    print("✓ Distribution plot saved")


def categorical_analysis():
    print("\nCategorical Analysis ...\n")
    # How many unique countries and regions?

    n_countries = df['Country'].nunique()
    n_regions = df['Region'].nunique()
    print(f"\nCountry: {n_countries} unique values")
    print(f"Region:  {n_regions} unique values")

    # Regions per country — to check if modeling will be country-confounded
    regions_per_country = df.groupby('Country')['Region'].count().sort_values(ascending=False)
    print("\nTop 10 countries by region count:")
    print(regions_per_country.head(10))


def missing_values_analysis():
    print("\nMissing Values Analysis ...\n")

    missing_pct = df.isnull().mean() * 100

    print("\n=== MISSING VALUES (%) ===")
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"{col}: {pct:.2f}% missing")

    missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    missing_df.plot(kind='bar', ax=ax, color=PALETTE['red'], edgecolor='none')
    ax.set_title('Missing Values by Column (%)', color=PALETTE['navy'])
    ax.set_ylabel('% Missing', color=PALETTE['gray'])
    ax.tick_params(axis='x', rotation=45, colors=PALETTE['gray'])
    ax.tick_params(axis='y', colors=PALETTE['gray'])

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, '02_missingness.png'), dpi=150)
    plt.show()

    # Test if missingness correlates with life satisfaction (MNAR check)
    target = 'Life Satisfaction (0–10)'
    print("\n=== MISSINGNESS vs LIFE SATISFACTION ===")
    for col in numeric_cols:
        if df[col].isnull().any() and col != target:
            present = df.loc[df[col].notna(), target].median()
            missing = df.loc[df[col].isna(), target].median()
            diff = present - missing
            if abs(diff) > 0.3:  # notable difference
                print(f"{col}: present median={present:.2f}, missing median={missing:.2f}")


def correlation_matrix_pearson_heatmap():
    print("\nCorrelation Matrix Pearson Heatmap ...")

    # Pearson correlation matrix for all numeric columns
    corr = df[numeric_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(14, 12))

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_heatmap",
        [PALETTE['red'], PALETTE['offwhite'], PALETTE['green']]
    )

    sns.heatmap(corr, annot=True, fmt='.2f',
                ax=ax, cmap=custom_cmap, center=0, vmin=-1,
                vmax=1, annot_kws={'size': 7, 'color': PALETTE['navy']},
                linewidths=0.3
                )

    ax.set_title(
        'Correlation Matrix of Well-Being Indicators',
        fontsize=14,
        color=PALETTE['navy']
    )

    ax.tick_params(colors=PALETTE['gray'])

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, '03_correlation_matrix.png'), dpi=150)
    plt.show()

    # Print top correlations with the target
    target_corr = corr['Life Satisfaction (0–10)'].drop('Life Satisfaction (0–10)'). \
        abs().sort_values(ascending=False)
    print("\nCorrelations with Life Satisfaction (absolute, ranked):")
    print(target_corr.round(2))


def eda():
    print(f'\n{"─" * 55} EDA STARTING {"─" * 55}\n')

    kda_analysis()
    categorical_analysis()
    missing_values_analysis()
    correlation_matrix_pearson_heatmap()

    print(f'\n{"─" * 55} EDA END {"─" * 55}\n')
