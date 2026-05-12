import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
from main import logging
from main import Fore, Style, init

from configs.config_paths_and_params import MODELS_DIR, OUTPUTS_DIR, TEST_METRICS_PATH, DATA_PROCESSED_X_TEST_FINAL, \
    DATA_PROCESSED_Y_TEST, DATA_PROCESSED_X_TRAINED_FINAL, DATA_PROCESSED_Y_TRAIN, TRAIN_METRICS_PATH, \
    PLOT_07_TRAIN_VS_TEST_R2_PATH, PLOT_08_ACTUAL_VS_PREDICTED_PATH

X_test = pd.read_csv(DATA_PROCESSED_X_TEST_FINAL)
y_test = pd.read_csv(DATA_PROCESSED_Y_TEST).squeeze()

# Load trained models
model_names = ['Random_Forest', 'Ridge_Regression', 'Gradient_Boosting']
models = {n: joblib.load(f'{MODELS_DIR}/{n}.pkl') for n in model_names}


# Train vs Test R² grouped bar chart
def trains_vs_test_R2():
    logging.info(f'Train vs Test R² grouped bar chart ...')
    print(f"\n{Fore.CYAN}Train vs Test R² grouped bar chart ...\n {Style.RESET_ALL}")

    df_test = pd.read_csv(TEST_METRICS_PATH)

    df_test['Model'] = (
        df_test['Model']
        .str.strip()
        .str.replace('_', ' ')
    )

    test_r2 = dict(zip(df_test['Model'], df_test['Test R²']))
    print(test_r2)

    df_train = pd.read_csv(TRAIN_METRICS_PATH)

    df_train['Model'] = (
        df_train['Model']
        .str.strip()
        .str.replace('_', ' ')
    )

    train_r2 = dict(zip(df_train['Model'], df_train['Train R²']))
    print(train_r2)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(train_r2))
    width = 0.35
    ax.bar(x_pos - width / 2, train_r2.values(), width, label='Train R²', color='#1e3a5f') # Blue
    ax.bar(x_pos + width / 2, [test_r2.get(k, 0) for k in train_r2], width,
           label='Test R²', color='#8b1a4a') # Red
    ax.set_xticks(x_pos)
    ax.set_xticklabels(train_r2.keys())
    ax.set_ylabel('R² Score')
    ax.set_title('Model R² Scores: Train vs Test')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_07_TRAIN_VS_TEST_R2_PATH, dpi=150)
    plt.show()

    print(f"{Fore.GREEN}\n✓ Train vs Test R² grouped bar chart plot saved {PLOT_07_TRAIN_VS_TEST_R2_PATH} {Style.RESET_ALL} ")
    logging.info(f'Train vs Test R² grouped bar chart plot saved {PLOT_07_TRAIN_VS_TEST_R2_PATH}')


# Actual vs Predicted: Random Forest
def actual_vs_predicted():
    logging.info(f'Actual vs Predicted: Random Forest plot ...')
    print(f"\n{Fore.CYAN}Actual vs Predicted: Random Forest plot  ...\n {Style.RESET_ALL}")

    best_model = models['Random_Forest']
    y_pred_rf = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred_rf, alpha=0.6, color='#1e3a5f', s=40)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
    ax.set_xlabel('Actual Life Satisfaction')
    ax.set_ylabel('Predicted Life Satisfaction')
    ax.set_title('Random Forest: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(PLOT_08_ACTUAL_VS_PREDICTED_PATH, dpi=150)
    plt.show()

    print(f"{Fore.GREEN}✓ Actual vs Predicted: Random Forest plot saved {PLOT_08_ACTUAL_VS_PREDICTED_PATH} {Style.RESET_ALL} ")
    logging.info(f'Actual vs Predicted: Random Forest plot saved {PLOT_08_ACTUAL_VS_PREDICTED_PATH}')


def models_analysis_main():
    logging.info(f'8. MODELS ANALYSIS')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}8. MODELS ANALYSIS {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    trains_vs_test_R2()

    actual_vs_predicted()


    logging.info(f'MODELS ANALYZED\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ MODELS ANALYZED {Style.RESET_ALL} {"─" * 20}\n')
