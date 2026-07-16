from configs.config_paths_and_params import LOG_PATH, LOGS_DIR
import os

os.makedirs(LOGS_DIR, exist_ok=True)
import logging
from colorama import Fore, Style, init

logging.basicConfig(
    filename=LOG_PATH,
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
import functools
import logging
import time

logger = logging.getLogger(__name__)


def log_calls(new_line: bool = False):
    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            logger.info(f"Entering {fn.__name__}")

            start = time.perf_counter()

            try:
                result = fn(*args, **kwargs)
                line = '\n' if new_line == True else ''
                logger.info(f"Finished {fn.__name__} in {time.perf_counter() - start} seconds{line}")

                return result

            except Exception as exc:
                logger.exception(f"{fn.__name__} failed with {exc}")
                raise

        return wrapper

    return decorator


@log_calls()
def test_setup():
    # logging.info(f'1. SETUP TESTING STARTING')
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

    # logging.info(f'SETUP TESTING END\n')
    print(f'\n {Fore.LIGHTGREEN_EX}✓ SETUP TESTING DONE {Style.RESET_ALL} {"─" * 20}\n')
