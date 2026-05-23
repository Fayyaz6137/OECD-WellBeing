import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib, os

from configs.config_paths_and_params import MODELS_DIR, DATA_PROCESSED_Y_TRAIN, DATA_PROCESSED_X_TRAINED_FINAL, \
    OUTPUTS_DIR, \
    PLOTS_DIR, \
    PALETTE, RANDOM_FOREST_MODEL_PATH, TRAIN_METRICS_PATH, PLOT_06_RF_IMPORTANCE_TRAINED_PATH
from main import logging
from main import Fore, Style, init

X_train = pd.read_csv(DATA_PROCESSED_X_TRAINED_FINAL)
y_train = pd.read_csv(DATA_PROCESSED_Y_TRAIN).squeeze()
models = None


def train_models():
    logging.info(f'Training Models ...')
    print(f"\n{Fore.CYAN}Training Models ...\n {Style.RESET_ALL}")

    global models
    # Model Definitions
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }

    train_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)

        r2 = r2_score(y_train, y_pred_train)
        mae = mean_absolute_error(y_train, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

        train_results.append({'Model': name, 'Train R²': r2,
                              'Train MAE': mae, 'Train RMSE': rmse})

        # Save  model
        joblib.dump(model, f'{MODELS_DIR}/{name.replace(" ", "_")}.pkl')

        print(f"{name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        logging.info(f'{name.replace(" ", "_")} Model trained and saved. {MODELS_DIR}/{name.replace(" ", "_")}.pkl')
        print(
            f"{Fore.GREEN}✓ {name.replace(" ", "_")} Model trained and saved. {MODELS_DIR}\\{name.replace(" ", "_")}.pkl {Style.RESET_ALL}\n ")

    df_train_results = pd.DataFrame(train_results)
    df_train_results.to_csv(f'{TRAIN_METRICS_PATH}', index=False)

    logging.info(f'Train Metrics saved. {TRAIN_METRICS_PATH}')

    print(df_train_results.to_string(index=False))

    print(f"{Fore.GREEN}\n✓ Train Metrics saved. {TRAIN_METRICS_PATH}' {Style.RESET_ALL} ")


# Feature Importance Visualization from Random Forest
def feature_imp_viz():
    logging.info(f'Feature Importance Visualization from Random Forest ...')
    print(f"\n{Fore.CYAN}Feature Importance Visualization from Random Forest ...\n {Style.RESET_ALL}")

    rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
    base_features = [c for c in X_train.columns if not c.endswith('_nan')]

    imp = pd.Series(
        rf_model.feature_importances_[:len(base_features)],
        index=base_features
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    imp.plot(
        kind='barh',
        ax=ax,
        color='#1e3a5f'
    )

    ax.invert_yaxis()

    ax.set_title(
        'Most Influential Predictors of Life Satisfaction',
        fontsize=14,
        color=PALETTE['navy']
    )

    ax.set_xlabel(
        'Relative Importance Score',
        color=PALETTE['gray']
    )

    ax.tick_params(axis='x', colors=PALETTE['gray'])
    ax.tick_params(axis='y', colors=PALETTE['gray'])

    plt.tight_layout()
    plt.savefig(PLOT_06_RF_IMPORTANCE_TRAINED_PATH, dpi=150)
    plt.show()

    logging.info(f'Random Forest Feature Importance Trained plot saved {PLOT_06_RF_IMPORTANCE_TRAINED_PATH}')
    print(
        f"{Fore.GREEN}\n✓ Random Forest Feature Importance Trained plot saved {PLOT_06_RF_IMPORTANCE_TRAINED_PATH} {Style.RESET_ALL} ")


def models_training_main():
    logging.info(f'6. MODELS TRAINING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}6. MODELS TRAINING {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    train_models()
    feature_imp_viz()

    logging.info(f'MODELS TRAINED\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ MODELS TRAINED AND SAVED {Style.RESET_ALL} {"─" * 20}\n')
