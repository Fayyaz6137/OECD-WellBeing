import os
import numpy as np
import pandas as pd
from configs.config import DATA_PROCESSED_CSV_PATH, DATA_PROCESSED_DIR, COLUMN_RENAME_MAPPING
from main import Fore, Style, init
from main import logging


def drop_unwanted_columns_rows(df):
    logging.info(f'"Removing unwanted columns and rows ...')
    print(f"\n{Fore.CYAN}Removing unwanted columns and rows ...\n {Style.RESET_ALL}")

    # Drop the unnamed first column (row numbering artifact)
    df = df.loc[:, ~df.columns.str.startswith(
        ('Unnamed', 'Satisfaction with housing affordability')
    )]

    # Drop the first row garbage value
    df = df.iloc[1:]

    print("Shape after loading:", df.shape)
    logging.info(f'Shape after loading: {df.shape}')

    print("Columns:", df.columns.tolist())
    logging.info(f'Columns: {df.columns.tolist()}')

    return df


def rename_columns(df):
    logging.info(f'"Renaming columns ...')
    print(f"\n{Fore.CYAN}Renaming columns ...\n {Style.RESET_ALL}")

    # Rename to semantically clear names with units
    df = df.rename(columns=COLUMN_RENAME_MAPPING)

    df_cleaned = df.copy()

    print("Renamed columns:", df_cleaned.columns.tolist())
    logging.info(f"Renamed columns: {df_cleaned.columns.tolist()}")
    # print(df_cleaned.head())
    #
    return df_cleaned


def clean_and_pre_process(df):
    logging.info(f'Cleaning and Pre-processing ...')
    print(f"\n{Fore.CYAN}Cleaning and Pre-processing ...\n {Style.RESET_ALL}")

    # Columns that need numeric conversion
    numeric_cols = [c for c in df.columns
                    if c not in ['Country', 'Region', 'Code']]

    for col in numeric_cols:
        df[col] = df[col].astype(str)

        # Remove percentage signs and thousands commas
        df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)

        # Replace OECD placeholder ".." and text annotations with NaN
        df[col] = df[col].replace({
            '..': np.nan,
            'Self assessment of life satisfaction': np.nan,
            '': np.nan,
            'nan': np.nan
        })

        # Cast to float, coercing remaining non-numeric entries to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"{Fore.LIGHTGREEN_EX}✓ Conversion complete. Dtypes:!{Style.RESET_ALL}")
    print(df.dtypes)
    logging.info(f'Conversion complete. Dtypes:\n{df.dtypes}')

    return df


def missing_check(df):
    logging.info(f'Checking Missing Values Percentage ...')
    print(f"\n{Fore.CYAN}Checking Missing Values Percentage ...\n {Style.RESET_ALL}")

    missing_pct = df.isnull().mean() * 100

    print("\nMissingness (%):")
    print(missing_pct.round(2))
    logging.info(f'Missingness (%):{missing_pct.round(2)}')


def data_cleaning_and_preperation(df):
    logging.info(f'3. DATA PREPROCESSING STARTING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}3. DATA PREPROCESSING {Style.RESET_ALL} ')
    print(f'{"─" * 200}\n')

    df = drop_unwanted_columns_rows(df)
    df = rename_columns(df)
    df = clean_and_pre_process(df)
    missing_check(df)

    print(f'\n\n{df.head()}')
    logging.info(f'\n{df.head()}')

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(DATA_PROCESSED_CSV_PATH, index=False)

    print(f"{Fore.GREEN}\n✓ Cleaned data saved to {DATA_PROCESSED_CSV_PATH}{Style.RESET_ALL}")
    logging.info(f'Cleaned data saved to {DATA_PROCESSED_CSV_PATH}')

    logging.info(f'DATA PREPROCESSING END\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ DATA PREPROCESSING DONE {Style.RESET_ALL} {"─" * 20}\n')
