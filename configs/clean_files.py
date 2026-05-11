import os
import shutil
from debugpy.launcher import output
from configs.config import CLEAN_FILES
from main import logging
from main import Fore, Style, init


def clean_files():
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}0. REMOVING OLD FILES {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')
    logging.info(f'0. REMOVING OLD FILES')

    for file in CLEAN_FILES:
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)

                print(f"{Fore.CYAN}{file} Removed{Style.RESET_ALL}")
                logging.info(f"{file} Removed")
            else:
                shutil.rmtree(file)

                print(f"{Fore.CYAN}{file} Removed{Style.RESET_ALL}")
                logging.info(f"{file} Removed")

    logging.info(f"OLD FILES REMOVED\n")
    print(f'\n{Fore.LIGHTGREEN_EX}✓ OLD FILES REMOVED {Style.RESET_ALL} {"─" * 20}\n')
