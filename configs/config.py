import os

from matplotlib import pyplot as plt

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


# ---------------------------- SWITCHES ------------------------------- #
DEBUG_SWITCH = 0  # DEBUG SWITCH

# ---------------------------- PATHS ------------------------------- #

# --- DATA ---
DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, 'OECD-Regional-Well-Being-Data-File.xlsx')

DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, 'oecd_cleaned.csv')

# --- RESULTS ---
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
CLEAN_FILES=[DATA_PROCESSED_DIR,RESULTS_DIR]

# ---------------------------- STYLE ------------------------------- #
PALETTE = {
    'navy': '#1E3A5F',
    'blue': '#2563EB',
    'teal': '#0891B2',
    'sky': '#7DD3FC',
    'green': '#059669',
    'red': '#DC2626',
    'amber': '#D97706',
    'gray': '#64748B',
    'lgray': '#CBD5E1',
    'offwhite': '#F8FAFC',
}

LINE_COLOURS = ['#0891B2', '#DC2626', '#059669', '#D97706', '#7C3AED']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#E2E8F0',
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
})
