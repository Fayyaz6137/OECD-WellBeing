import os

import numpy as np
import pandas as pd

from configs.config import DATA_PROCESSED_CSV_PATH, DATA_PROCESSED_DIR, COLUMN_RENAME_MAPPING


def drop_unwanted_columns_rows(df):
    print("\nRemoving unwanted columns and rows ...\n")
    # Drop the unnamed first column (row numbering artifact)
    df = df.loc[:, ~df.columns.str.startswith(
        ('Unnamed', 'Satisfaction with housing affordability')
    )]

    # Drop the first row garbage value
    df = df.iloc[1:]

    print("Shape after loading:", df.shape)
    print("Columns:", df.columns.tolist())

    return df


def rename_columns(df):
    print("\nRenaming columns ...\n")

    # Rename to semantically clear names with units
    df = df.rename(columns=COLUMN_RENAME_MAPPING)

    df_cleaned = df.copy()
    print("Renamed columns:", df_cleaned.columns.tolist())
    # print(df_cleaned.head())
    #
    return df_cleaned


def clean_and_pre_process(df):
    print("\nCleaning and Pre-processing ...\n")
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

    print("Conversion complete. Dtypes:")
    print(df.dtypes)
    return df


def missing_check(df):
    print("\nChecking Missing Values Percentage ...\n")
    missing_pct = df.isnull().mean() * 100
    print("\nMissingness (%):")
    print(missing_pct.round(2))


def data_cleaning_and_preperation(df):
    print(f'\n{"─" * 55} DATA PREPROCESSING STARTING {"─" * 55}\n')

    df = drop_unwanted_columns_rows(df)
    df = rename_columns(df)
    df = clean_and_pre_process(df)
    missing_check(df)

    print(f'\n\n{df.head()}')

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(DATA_PROCESSED_CSV_PATH, index=False)
    print(f"\n✓ Cleaned data saved to {DATA_PROCESSED_CSV_PATH}")

    print(f'\n{"─" * 55} DATA PREPROCESSING END {"─" * 55}\n')
