import os
import shutil

from debugpy.launcher import output

from configs.config import CLEAN_FILES


def clean_files():
    for file in CLEAN_FILES:
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
                print(f"{file} Removed")
            else:
                shutil.rmtree(file)
                print(f"{file} Removed")
