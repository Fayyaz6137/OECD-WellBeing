import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
from main import logging
from main import Fore, Style, init

from configs.config_paths_and_params import MODELS_DIR, OUTPUTS_DIR, TEST_METRICS_PATH, DATA_PROCESSED_X_TEST_FINAL, \
    DATA_PROCESSED_Y_TEST, DATA_PROCESSED_X_TRAINED_FINAL, DATA_PROCESSED_Y_TRAIN

X_test = pd.read_csv(DATA_PROCESSED_X_TEST_FINAL)
y_test = pd.read_csv(DATA_PROCESSED_Y_TEST).squeeze()
X_train = pd.read_csv(DATA_PROCESSED_X_TRAINED_FINAL)
y_train = pd.read_csv(DATA_PROCESSED_Y_TRAIN).squeeze()

# Load trained models
model_names = ['Random_Forest', 'Ridge_Regression', 'Gradient_Boosting']
models = {n: joblib.load(f'{MODELS_DIR}/{n}.pkl') for n in model_names}


# Ridge hyperparameter tuning via GridSearchCV
def ridge_tuning():
    logging.info(f'Ridge Tuning ...')
    print(f"\n{Fore.CYAN}Ridge Tuning ...\n {Style.RESET_ALL}")

    global models

    ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
    ridge_cv.fit(X_train, y_train)
    best_alpha = ridge_cv.best_params_['alpha']
    print(f"Best Ridge alpha: {best_alpha}")

    ridge_tuned = Ridge(alpha=best_alpha)
    ridge_tuned.fit(X_train, y_train)
    models['Ridge_Tuned'] = ridge_tuned


# Evaluate on test set
def evaluate_on_test_set():
    logging.info(f'Evaluating on test set ...')
    print(f"\n{Fore.CYAN}Evaluating on test set ...\n {Style.RESET_ALL}")

    test_results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        test_results.append({
            'Model': name,
            'Test R²': r2_score(y_test, y_pred),
            'Test MAE': mean_absolute_error(y_test, y_pred),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        })

    df_test = pd.DataFrame(test_results)
    return df_test


def models_testing_main():
    logging.info(f'7. MODELS TESTING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}7. MODELS TESTING {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    ridge_tuning()
    df_test = evaluate_on_test_set()
    df_test.to_csv(f'{TEST_METRICS_PATH}', index=False)

    logging.info(f'Test Metrics saved. {TEST_METRICS_PATH}')

    print("\n=== TEST SET PERFORMANCE ===")
    print(df_test.to_string(index=False))

    print(f"{Fore.GREEN}\n✓ Test Metrics saved. {TEST_METRICS_PATH} {Style.RESET_ALL} ")

    logging.info(f'MODELS TESTED\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ MODELS TESTED {Style.RESET_ALL} {"─" * 20}\n')
