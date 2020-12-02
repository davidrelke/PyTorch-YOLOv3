import os
from os import listdir
from os.path import isfile, join
from typing import List

labels_path = os.path.abspath("labels")

label_files: List[str] = [labels_path + "/" + f for f in listdir(labels_path) if isfile(join(labels_path, f))]

for file in label_files:
    new_lines: List[str] = []
    with open(file, "r") as file_carla:
        for line in file_carla.readlines():
            if line.startswith("0"):
                new_lines.append(line)
    with open(file, "w") as new_file:
        for line in new_lines:
            new_file.write(f"{line}")