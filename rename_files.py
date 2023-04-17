import os
from pathlib import Path

# define the directory path
directory_path = "./raw_faces"

# loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".heic"):
        # create a Path object for the file
        filepath = Path(directory_path) / filename
        # change the file extension to .jpeg
        new_filepath = filepath.with_suffix(".jpeg")
        # rename the file
        os.rename(filepath, new_filepath)
        print(f"Renamed {filename} to {new_filepath.name}")
