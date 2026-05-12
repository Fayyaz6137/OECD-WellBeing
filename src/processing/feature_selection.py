import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression

from configs.config_paths_and_params import DATA_PROCESSED_X_TRAINED_SCALED, DATA_PROCESSED_Y_TRAIN, PLOTS_DIR, PALETTE, \
    DATA_PROCESSED_X_TEST_SCALED, DATA_PROCESSED_X_TRAINED_FINAL, DATA_PROCESSED_X_TEST_FINAL, RESULTS_DIR, \
    DATA_PROCESSED_DIR, PLOT_04_RF_IMPORTANCE_PATH, PLOT_05_MUTUAL_INFO_SCORE_PATH
from main import logging
from main import Fore, Style, init

X_train = pd.read_csv(DATA_PROCESSED_X_TRAINED_SCALED)
y_train = pd.read_csv(DATA_PROCESSED_Y_TRAIN).squeeze()
X_test = pd.read_csv(DATA_PROCESSED_X_TEST_SCALED).squeeze()
base_cols = [c for c in X_train.columns if not c.endswith('_nan')]

rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_train[base_cols], y_train)

importances = pd.Series(
    rf_selector.feature_importances_,
    index=base_cols
).sort_values(ascending=False)

X_train_final = None
X_test_final = None


# Random Forest Feature Importance
def rf_feat_imp():
    logging.info(f'Random Forest Feature Importance ...')
    print(f"\n{Fore.CYAN}Random Forest Feature Importance ...\n {Style.RESET_ALL}")

    # Exclude the _nan indicator columns from importance analysis (but keep for modelling)
    fig, ax = plt.subplots(figsize=(10, 6))

    importances.plot(kind='barh', ax=ax, color='#1e3a5f')

    ax.invert_yaxis()
    ax.set_title('Feature Importance from Random Forest', color=PALETTE['navy'], fontsize=14)
    ax.set_xlabel('Importance Score', color=PALETTE['gray'])
    ax.tick_params(axis='x', colors=PALETTE['gray'])
    ax.tick_params(axis='y', colors=PALETTE['gray'])
    plt.tight_layout()

    plt.savefig(PLOT_04_RF_IMPORTANCE_PATH, dpi=150)
    plt.show()

    print(
        f"{Fore.GREEN}\n✓ Random Forest Feature Importance plot saved {PLOT_04_RF_IMPORTANCE_PATH} {Style.RESET_ALL} ")
    logging.info(f'Random Forest Feature Importance plot saved {PLOT_04_RF_IMPORTANCE_PATH}')

    print("\nRandom Forest Feature Importances:")
    print(importances.round(4))
    logging.info(f'Random Forest Feature Importances:\n{importances.round(4)}')


def mutual_info_score():
    logging.info(f'Mutual Info Score ...')
    print(f"\n{Fore.CYAN}Mutual Info Score ...\n {Style.RESET_ALL}")

    # Mutual information captures non-linear feature–target dependencies
    mi_scores = mutual_info_regression(X_train[base_cols], y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=base_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    mi_series.plot(kind='barh', ax=ax, color='#1e3a5f')
    ax.invert_yaxis()
    ax.set_title('Mutual Information Scores', fontsize=14, color=PALETTE['navy'])
    ax.set_xlabel('Mutual Information with Life Satisfaction', color=PALETTE['gray'])
    ax.tick_params(axis='x', colors=PALETTE['gray'])
    ax.tick_params(axis='y', colors=PALETTE['gray'])

    plt.tight_layout()
    plt.savefig(PLOT_05_MUTUAL_INFO_SCORE_PATH, dpi=150)
    plt.show()

    print(f"{Fore.GREEN}\n✓ Mutual Information Scores plot saved {PLOT_05_MUTUAL_INFO_SCORE_PATH} {Style.RESET_ALL} ")
    logging.info(f'Mutual Information Scores plot saved {PLOT_05_MUTUAL_INFO_SCORE_PATH}')

    print("\nMutual Information Scores:")
    print(mi_series.round(4))
    logging.info(f'Mutual Information Scores:\n{mi_series.round(4)}')


# Recursive Feature Elimination (RFE) and final subset selection
def rfe():
    logging.info(f'Recursive Feature Elimination ...')
    print(f"\n{Fore.CYAN}Recursive Feature Elimination ...\n {Style.RESET_ALL}")

    global X_train, X_test, X_train_final, X_test_final

    # RFE with Linear Regression estimator, selecting top 10 features
    rfe = RFE(LinearRegression(), n_features_to_select=10)
    rfe.fit(X_train[base_cols], y_train)

    rfe_selected = [c for c, s in zip(base_cols, rfe.support_) if s]
    print("\nRFE selected features:", rfe_selected)

    # Final selection: features above median importance in Random Forest
    median_importance = importances.median()
    final_features = importances[importances >= median_importance].index.tolist()
    print("\nFinal selected features (RF importance ≥ median):")
    print(final_features)

    # Save selected feature names
    with open(os.path.join(RESULTS_DIR, 'selected_features.txt'), 'w') as f:
        f.write('\n'.join(final_features))

    # Create final model ready datasets using only selected features + _nan flags
    nan_flags = [c for c in X_train.columns if c.endswith('_nan')]
    model_cols = final_features + nan_flags

    X_train_final = X_train[model_cols]
    X_test_final = X_test[model_cols]


def feature_selection_main():
    rf_feat_imp()
    mutual_info_score()
    rfe()

    X_train_final.to_csv(DATA_PROCESSED_X_TRAINED_FINAL, index=False)
    X_test_final.to_csv(DATA_PROCESSED_X_TEST_FINAL, index=False)

    print(f"{Fore.GREEN}\n✓ Feature selection complete. Final datasets saved. {DATA_PROCESSED_DIR} {Style.RESET_ALL} ")
    logging.info(f'Feature selection complete. Final datasets saved. {DATA_PROCESSED_DIR}')
