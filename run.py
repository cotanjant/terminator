from pathlib import Path

from src import main
import os

SRC_PATH = Path("src")

if SRC_PATH.is_dir():
    print("Starting app...")
    if __name__ == "__main__":
        main()
else:
    print("src/ dir not found")
