import os
import pandas as pd

import requests

from configs.config import DATA_RAW_EXCEL_PATH


def get_raw_data():
    print(f'\n{"─" * 55} DATA RETRIEVAL STARTING {"─" * 55}\n')
    # Load all sheet names to understand file structure
    xl = pd.ExcelFile(DATA_RAW_EXCEL_PATH)
    print("Available sheets:", xl.sheet_names)

    # Preview the main integrated indicator sheet
    df_raw = pd.read_excel(DATA_RAW_EXCEL_PATH,
                           sheet_name='Indicator_Last',
                           header=6)  # skip 6 metadata rows at the top
    print("\nShape:", df_raw.shape)
    print("\nFirst 5 rows:")
    print(df_raw.head())
    print("\nColumns:")
    print(df_raw.columns.tolist())
    print(f'\n{"─" * 55} DATA RETRIEVAL END {"─" * 55}\n')
    return df_raw
