import pandas as pd

from configs.config_paths_and_params import LOG_PATH, LOGS_DIR, TRAIN_METRICS_PATH
import os



os.makedirs(LOGS_DIR, exist_ok=True)
import logging

logging.basicConfig(
    filename=LOG_PATH,
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO  # set to INFO for verbose gensim output
)

from colorama import Fore, Style, init

def main():
    logging.info(f'START')

    debug = 1
    if debug:
        # ---------------------------- 0. Remove Old Data ------------------------------- #
        from configs import config_dirs
        config_dirs.config_dirs_main()

        # ---------------------------- 1. Test Setup ------------------------------- #
        from configs import config_setup
        config_setup.test_setup()

        # ---------------------------- 2. Get Data ------------------------------- #
        from src.data import fetch_data
        df_raw = fetch_data.get_raw_data()

        # ---------------------------- 3. Pre-process Data ------------------------------- #
        from src.data import pre_process_data
        pre_process_data.data_pre_processing(df_raw)

        # ---------------------------- 4.EDA ------------------------------- #
        from src.data import exp_data_analysis
        exp_data_analysis.eda()

        # ---------------------------- 5. Processing ------------------------------- #
        from src.processing import process
        process.process_main()

        # ---------------------------- 6. Models Training ------------------------------- #
        from src.models import models_training
        models_training.models_training_main()

        # ---------------------------- 7. Models Testing ------------------------------- #
        from src.models import models_testing
        models_testing.models_testing_main()

        # ---------------------------- 8. Models Analysis ------------------------------- #
        from src.models import models_analysis
        models_analysis.models_analysis_main()

        # ---------------------------- 9. Final Analysis ------------------------------- #
        from src import final_results
        final_results.final_results_main()

    logging.info(f'\nEND\n')


if __name__ == "__main__":
    main()
