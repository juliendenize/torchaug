# =======================================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# License: CECILL-C
# =======================================================

"""Check if all files contain the required license header."""

import os
from pathlib import Path


FILE_PATH = Path(os.path.abspath(__file__))

# Header for each file type
HEADER = """# =======================================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# License: CECILL-C
# =======================================================
\n"""


def check_header(file_path, header):
    """Check if the file contains the required header."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        return content.startswith(header) or content == header[:-1]


def main():
    """Check if all files contain the required license header."""
    missing_headers = []
    folders = ["torchaug"]
    for folder in folders:
        for file in FILE_PATH.parent.glob(f"./{folder}/**/*.py"):
            if not check_header(file, HEADER):
                missing_headers.append(file)

    if missing_headers:
        print("The following files miss the required license header :")
        for file in missing_headers:
            print(file)
        exit(1)
    else:
        print("All files include the required license header.")


if __name__ == "__main__":
    main()
