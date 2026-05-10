import os
# ---------------------------- PARAMS ------------------------------- #
COLUMN_RENAME_MAPPING = {
        'Country': 'Country',
        'Region': 'Region',
        'Code': 'Code',

        'Disposable income per capita': 'Disposable Income (USD PPP)',
        'Employment rate': 'Employment Rate (%)',
        'Unemployment rate': 'Unemployment Rate (%)',

        'Homicide rate': 'Homicide Rate (per 100k)',
        'Life expectancy': 'Life Expectancy (years)',

        'Population with at least secondary education': 'Education (%)',

        'Number of rooms per capita': 'Rooms per Person',

        'Mortality rate': 'Mortality Rate (per 1k)',

        'Voter turnout': 'Voter Turnout (%)',

        'Households broadband access': 'Broadband Access (%)',

        'Air quality (PM2.5)': 'Air Pollution (PM2.5 µg/m³)',

        'Life satisfaction': 'Life Satisfaction (0–10)',

        'Internet speed': 'Internet Speed Deviation (%)',

        'Perceived social network support ': 'Social Support (%)'
    }

# ---------------------------- PATHS ------------------------------- #

# ── DATA ──
DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, 'OECD-Regional-Well-Being-Data-File.xlsx')


DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, 'oecd_cleaned.csv')
