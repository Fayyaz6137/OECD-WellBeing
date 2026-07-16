import os
import shutil
from debugpy.launcher import output
from configs.config_paths_and_params import REMOVE_OLD_DIRS, CREATE_DIRS
from main import logging
from main import Fore, Style, init
from configs.config_setup import log_calls

@log_calls()
def clean_dirs():
    # logging.info(f'0. REMOVING OLD DIRS')
    print(f"{Fore.CYAN}Removing Old Directories ... {Style.RESET_ALL}\n")

    for file in REMOVE_OLD_DIRS:
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)

                print(f"{file} -- Removed")
                # logging.info(f"{file} -- Removed")
            else:
                shutil.rmtree(file)

                print(f"{file} -- Removed")
                # logging.info(f"{file} -- Removed")

    # logging.info(f"Old Directories Removed\n")
    print(f"{Fore.GREEN}\n✓ Old Directories Removed \n")

@log_calls()
def create_dirs():
    # logging.info(f'CREATING NEW DIRECTORIES')
    print(f"{Fore.CYAN}Creating New Directories ... {Style.RESET_ALL}\n")

    for file in CREATE_DIRS:
        os.makedirs(file, exist_ok=True)

        print(f"{file} -- Created")
        # logging.info(f"{file} -- Created")

    # logging.info(f"New Directories Created\n")
    print(f"{Fore.GREEN}\n✓ New Directories Created ")

@log_calls(True)
def config_dirs_main():
    # logging.info(f'0. SETTING UP DIRECTORIES')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}0. SETTING UP DIRECTORIES {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    clean_dirs()
    create_dirs()

    # logging.info(f"DIRECTORIES ALL SET UP\n")
    print(f'\n{Fore.LIGHTGREEN_EX}✓ DIRECTORIES ALL SET UP {Style.RESET_ALL} {"─" * 20}\n')
