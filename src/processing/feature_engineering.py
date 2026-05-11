import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from configs.config import DATA_PROCESSED_CSV_PATH, DATA_PROCESSED_X_TRAINED_SCALED, DATA_PROCESSED_Y_TRAIN, \
    DATA_PROCESSED_X_TEST_SCALED, DATA_PROCESSED_Y_TEST, DATA_PROCESSED_DIR

from main import logging
from main import Fore, Style, init

X_train = None
X_test = None
y_train = None
y_test = None
meta_train = None
meta_test = None
num_cols = None
X_train_scaled = None
X_test_scaled = None

df = pd.read_csv(DATA_PROCESSED_CSV_PATH)


def train_and_test_split():
    logging.info(f'Train Test Split ...')
    print(f"\n{Fore.CYAN}Train Test Split ...\n {Style.RESET_ALL}")

    global X_train, X_test, y_train, y_test, meta_train, meta_test, num_cols, df

    # Clean the target variable — drop rows with missing life satisfaction
    target = 'Life Satisfaction (0–10)'
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target])
    print(f"Rows after dropping missing target: {len(df)}")

    # Define features and target
    cat_cols = ['Country', 'Region', 'Code']
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]

    X = df[num_cols]
    y = df[target]
    meta = df[cat_cols]  # keep for later interpretability

    # 80/20 split BEFORE any imputation (prevents data leakage)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.20, random_state=42
    )
    print(f"Train: {X_train.shape[0]} regions | Test: {X_test.shape[0]} regions")


def add_missing_indicators():
    logging.info(f'Adding Missing Indicators ...')
    print(f"\n{Fore.CYAN}Adding Missing Indicators ...\n {Style.RESET_ALL}")

    # Add a binary indicator column for each feature with missing values
    # This preserves the MNAR signal identified in EDA
    cols_with_nan = [c for c in num_cols if X_train[c].isnull().any()]

    for col in cols_with_nan:
        X_train[f'{col}_nan'] = X_train[col].isnull().astype(int)
        X_test[f'{col}_nan'] = X_test[col].isnull().astype(int)

    print(f"Added {len(cols_with_nan)} missingness indicator columns")


def median_imputation():
    logging.info(f'Median Imputation ...')
    print(f"\n{Fore.CYAN}Median Imputation ...\n {Style.RESET_ALL}")

    global X_train, X_test

    # Compute medians on TRAINING set only
    train_medians = X_train[num_cols].median()

    # Apply to both train and test
    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_test[num_cols] = X_test[num_cols].fillna(train_medians)

    print("Remaining NaN in train:", X_train.isnull().sum().sum())
    print("Remaining NaN in test: ", X_test.isnull().sum().sum())


def min_max_scaling():
    logging.info(f'Min-Max scaling ...')
    print(f"\n{Fore.CYAN}Min-Max scaling ...\n {Style.RESET_ALL}")

    global X_train, X_test, X_train_scaled, X_test_scaled

    # Scale all features to [0, 1] range
    # Fit scaler on training data ONLY to prevent leakage
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )


def feature_engineering_main():
    train_and_test_split()
    add_missing_indicators()
    median_imputation()
    min_max_scaling()

    # Save final datasets
    X_train_scaled.to_csv(DATA_PROCESSED_X_TRAINED_SCALED, index=False)
    X_test_scaled.to_csv(DATA_PROCESSED_X_TEST_SCALED, index=False)
    y_train.to_csv(DATA_PROCESSED_Y_TRAIN, index=False)
    y_test.to_csv(DATA_PROCESSED_Y_TEST, index=False)

    print(f"{Fore.GREEN}✓ Feature engineering complete. All files saved. {DATA_PROCESSED_DIR} {Style.RESET_ALL} ")
    logging.info(f'Feature engineering complete. All files saved. {DATA_PROCESSED_DIR} ')
