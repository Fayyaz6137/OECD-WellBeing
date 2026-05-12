import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import logging
from main import Fore, Style, init
from configs.config_paths_and_params import PLOT_09_GINI_BY_COUNTRY_PATH, PLOT_10_REGIONAL_BOXPLOT_PATH, \
    GINI_BY_COUNTRY_PATH

df = pd.read_csv('data/processed/oecd_cleaned.csv')
target = 'Life Satisfaction (0–10)'


# Compute Gini coefficient for a given array
def gini_coefficient(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    values = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(values)
    return (2 * np.sum((range(1, n + 1)) * values) - (n + 1) * cumvals[-1]) / \
        (n * cumvals[-1])


# Compute Gini per country (need at least 3 regions for meaningful result)
def gini_per_country():
    logging.info(f'Gini by Country Analysis ...')
    print(f"\n{Fore.CYAN}Gini by Country Analysis ...\n {Style.RESET_ALL}")

    gini_by_country = {}
    for country, group in df.groupby('Country'):
        scores = group[target].dropna()
        if len(scores) >= 3:
            gini_by_country[country] = gini_coefficient(scores)

    gini_series = pd.Series(gini_by_country).sort_values(ascending=False)
    gini_series.to_csv(GINI_BY_COUNTRY_PATH, header=True)

    print(
        f"{Fore.GREEN}\n✓ Gini by Country CSV saved {GINI_BY_COUNTRY_PATH} {Style.RESET_ALL} ")
    logging.info(f'Gini by Country CSV saved {GINI_BY_COUNTRY_PATH}')

    # Plot top 15 most unequal countries
    top15 = gini_series.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top15.sort_values().plot(kind='barh', ax=ax, color='#1e3a5f')
    ax.set_title('Top 15 Countries with Highest Regional Inequality in Life Satisfaction')
    ax.set_xlabel('Gini Coefficient')
    plt.tight_layout()
    plt.savefig(PLOT_09_GINI_BY_COUNTRY_PATH, dpi=150)
    plt.show()
    logging.info(f'Gini by Country plot saved {PLOT_09_GINI_BY_COUNTRY_PATH}')

    print("\nTop 15 countries with highest internal life satisfaction inequality:")
    print(top15.round(4))

    print(
        f"{Fore.GREEN}\n✓ Gini by Country plot saved {PLOT_09_GINI_BY_COUNTRY_PATH} {Style.RESET_ALL} ")


# Regional Life Satisfaction
def regeional_life_satisfaction_plot():
    logging.info(f'Regional Life Satisfaction Analysis ...')
    print(f"\n{Fore.CYAN}Regional Life Satisfaction Analysis ...\n {Style.RESET_ALL}")

    top12_countries = df.groupby('Country')[target].count().nlargest(12).index

    df_top12 = df[df['Country'].isin(top12_countries)]

    fig, ax = plt.subplots(figsize=(14, 6))
    groups = [df_top12[df_top12['Country'] == c][target].dropna()
              for c in top12_countries]
    ax.boxplot(groups, labels=top12_countries, patch_artist=True,
               boxprops=dict(facecolor='#1E3A5F', alpha=0.7),
               medianprops=dict(color='white', linewidth=2))
    ax.set_title('Regional Variation in Life Satisfaction (Top 12 Countries by Region Count)')
    ax.set_ylabel('Life Satisfaction (0–10)')
    ax.set_xlabel('Country')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_10_REGIONAL_BOXPLOT_PATH, dpi=150)
    plt.show()

    logging.info(f'Regional Life Satisfaction plot saved {PLOT_10_REGIONAL_BOXPLOT_PATH}')
    print(
        f"{Fore.GREEN}\n✓ Regional Life Satisfaction plot saved {PLOT_10_REGIONAL_BOXPLOT_PATH} {Style.RESET_ALL} ")


def final_results_main():
    logging.info(f'9. FINAL ANALYSIS')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}9. FINAL ANALYSIS {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    gini_per_country()
    regeional_life_satisfaction_plot()

    logging.info(f'FINAL ANALYSIS DONE\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ FINAL ANALYSIS DONE {Style.RESET_ALL} {"─" * 20}\n')
