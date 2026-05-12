import os
import pandas as pd
import requests
from configs.config_paths_and_params import DATA_RAW_EXCEL_PATH
from main import logging
from main import Fore, Style, init


def get_raw_data():
    logging.info(f'2. DATA RETRIEVAL STARTING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}2. DATA RETRIEVAL {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    # Load all sheet names to understand file structure
    xl = pd.ExcelFile(DATA_RAW_EXCEL_PATH)
    print("Available sheets:", xl.sheet_names)

    # Preview the main integrated indicator sheet
    df_raw = pd.read_excel(DATA_RAW_EXCEL_PATH,
                           sheet_name='Indicator_Last',
                           header=6)  # skip 6 metadata rows at the top

    # LOG the results
    print("\nShape:", df_raw.shape)
    logging.info(f"Shape:{df_raw.shape}")

    print("\nFirst 5 rows:")
    logging.info("First 5 rows:")

    print(df_raw.head())
    logging.info(f'\n{df_raw.head()}')

    print("\nColumns:")
    print(df_raw.columns.tolist())
    logging.info(f"Columns: {df_raw.columns.tolist()}")

    logging.info(f'DATA RETRIEVAL END\n')
    print(f'\n {Fore.LIGHTGREEN_EX}✓ DATA RETRIEVED {Style.RESET_ALL} {"─" * 20}\n')

    return df_raw
