import os

from configs.config import LOG_PATH, LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)
import logging

logging.basicConfig(
    filename=LOG_PATH,
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO  # set to INFO for verbose gensim output
)
from colorama import Fore, Style, init


# init()

def main():
    # print(f'\n{"─" * 55} {Fore.LIGHTGREEN_EX} START {Style.RESET_ALL} {"─" * 55}\n')
    logging.info(f'START')

    run_all = 1
    if run_all:
        # ---------------------------- 0. Remove Old Data -------------------------------
        from configs import clean_files
        clean_files.clean_files()

        # ---------------------------- 1. Test Setup ------------------------------- #
        from configs import test_setup

        test_setup.test_setup()

        # ---------------------------- 2. Get Data ------------------------------- #
        from src.data import fetch_data

        df_raw = fetch_data.get_raw_data()

        # ---------------------------- 3. Pre-process Data ------------------------------- #
        from src.data import pre_process_data

        pre_process_data.data_cleaning_and_preperation(df_raw)

        # ---------------------------- 4. EDA ------------------------------- #
        from src.anlysis import exp_data_analysis

        exp_data_analysis.eda()

        # ---------------------------- 5. Processing ------------------------------- #
        from src.processing import process

        process.process_main()

        logging.info(f'\nEND\n')


if __name__ == "__main__":
    main()
