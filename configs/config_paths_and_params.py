from datetime import datetime
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

DATA_PROCESSED_X_TRAINED_SCALED = os.path.join(DATA_PROCESSED_DIR, 'X_train_scaled.csv')
DATA_PROCESSED_X_TEST_SCALED = os.path.join(DATA_PROCESSED_DIR, 'X_test_scaled.csv')
DATA_PROCESSED_Y_TRAIN = os.path.join(DATA_PROCESSED_DIR, 'y_train.csv')
DATA_PROCESSED_Y_TEST = os.path.join(DATA_PROCESSED_DIR, 'y_test.csv')

DATA_PROCESSED_X_TRAINED_FINAL = os.path.join(DATA_PROCESSED_DIR, 'X_train_final.csv')
DATA_PROCESSED_X_TEST_FINAL = os.path.join(DATA_PROCESSED_DIR, 'X_test_final.csv')

# --- RESULTS ---
RESULTS_DIR = 'results'

SELECTED_FEATURES_DIR = os.path.join(RESULTS_DIR, 'selected_features')

OUTPUTS_DIR = os.path.join(RESULTS_DIR, 'outputs')
TEST_METRICS_PATH = os.path.join(OUTPUTS_DIR, 'test_metrics.csv')
TRAIN_METRICS_PATH = os.path.join(OUTPUTS_DIR, 'train_metrics.csv')
GINI_BY_COUNTRY_PATH = os.path.join(OUTPUTS_DIR, 'gini_by_country.csv')

# --- PLOTS ---
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

PLOT_01_DISTRIBUTION_PATH = os.path.join(PLOTS_DIR, '01_distributions.png')
PLOT_02_MISSINGNESS_PATH = os.path.join(PLOTS_DIR, '02_missingness.png')
PLOT_03_CORRELATION_MATRIX_PATH = os.path.join(PLOTS_DIR, '03_correlation_matrix.png')
PLOT_04_RF_IMPORTANCE_PATH = os.path.join(PLOTS_DIR, '04_rf_importance.png')
PLOT_05_MUTUAL_INFO_SCORE_PATH = os.path.join(PLOTS_DIR, '05_mutual_info.png')
PLOT_06_RF_IMPORTANCE_TRAINED_PATH = os.path.join(PLOTS_DIR, '06_rf_importance_trained.png')
PLOT_07_TRAIN_VS_TEST_R2_PATH = os.path.join(PLOTS_DIR, '07_train_vs_test.png')
PLOT_08_ACTUAL_VS_PREDICTED_PATH = os.path.join(PLOTS_DIR, '08_actual_vs_predicted.png')
PLOT_09_GINI_BY_COUNTRY_PATH = os.path.join(PLOTS_DIR, '09_gini_by_country.png')
PLOT_10_REGIONAL_BOXPLOT_PATH = os.path.join(PLOTS_DIR, '10_regional_boxplot.png')
PLOT_11_RESIDUALS_PATH = os.path.join(PLOTS_DIR, '11_residuals.png.png')

# --- MODELS ---
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODELS_DIR, 'Random_Forest.pkl')

# --- LOGS ---
LOGS_DIR = 'logs'
LOG_PATH = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

REMOVE_OLD_DIRS = [DATA_PROCESSED_DIR, RESULTS_DIR]
CREATE_DIRS = [DATA_PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, OUTPUTS_DIR, MODELS_DIR]

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
