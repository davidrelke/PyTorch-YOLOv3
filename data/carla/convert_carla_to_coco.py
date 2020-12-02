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
            line_split = line.split(" ")
            if line_split[0] == "1":
                line_split[0] = "2"
            elif line_split[0] == "2":
                line_split[0] = "7"
            elif line_split[0] == "4":
                line_split[0] = "1"
            elif line_split[0] == "5":
                line_split[0] = "0"

            new_lines.append(" ".join(line_split))

    with open(file, "w") as new_file:
        for line in new_lines:
            new_file.write(f"{line}")