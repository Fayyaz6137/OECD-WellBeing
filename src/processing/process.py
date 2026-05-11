from main import logging
from main import Fore, Style, init


def process_main():
    logging.info(f'5. DATA PROCESSING')
    print(f'\n{"─" * 200}')
    print(f'{" " * 55} {Fore.LIGHTGREEN_EX}5. DATA PROCESSING {Style.RESET_ALL}')
    print(f'{"─" * 200}\n')

    logging.info(f'Feature Engineering ...')
    print(f"\n{Fore.LIGHTGREEN_EX}Feature Engineering ... {Style.RESET_ALL}")

    from src.processing import feature_engineering
    feature_engineering.feature_engineering_main()

    logging.info(f'Feature Selection ...')
    print(f"\n\n{Fore.LIGHTGREEN_EX}Feature Selection ... {Style.RESET_ALL}")

    from src.processing import feature_selection
    feature_selection.feature_selection_main()

    logging.info(f'DATA PROCESSING END\n')
    print(f'\n{Fore.LIGHTGREEN_EX}✓ DATA PROCESSING DONE {Style.RESET_ALL} {"─" * 20}\n')
