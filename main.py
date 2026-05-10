


def main():
    print(f'\n{"─" * 55} START {"─" * 55}\n')

    # ---------------------------- 0. Remove Old Data ------------------------------- #
    print("\nRemoving Old Files ...\n")
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


if __name__ == "__main__":
    main()
