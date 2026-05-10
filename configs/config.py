import os

# ---------------------------- PATHS ------------------------------- #

# ── DATA ──
DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, 'OECD-Regional-Well-Being-Data-File.xlsx')


DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, 'oecd_cleaned.csv')
