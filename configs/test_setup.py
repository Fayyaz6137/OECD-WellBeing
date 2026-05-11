from main import logging
from main import Fore, Style, init


def test_setup():
    logging.info(f'1. SETUP TESTING STARTING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}1. SETUP TEST {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.feature_selection import mutual_info_regression, RFE
    from sklearn.linear_model import LinearRegression

    print(f"{Fore.CYAN} pandas {pd.__version__}, sklearn available {Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ All imports successful!{Style.RESET_ALL}")

    logging.info(f'SETUP TESTING END\n')
    print(f'\n {Fore.LIGHTGREEN_EX}✓ SETUP TESTING DONE {Style.RESET_ALL} {"─" * 20}\n')
