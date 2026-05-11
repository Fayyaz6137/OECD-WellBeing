import pandas as pd

from configs.config import OUTPUTS_DIR


def models_analysis_main():
    df = pd.read_csv(f'{OUTPUTS_DIR}/test_metrics.csv')

    train_r2 = dict(zip(df['Model'], df['Test R2']))

    print(train_r2)
