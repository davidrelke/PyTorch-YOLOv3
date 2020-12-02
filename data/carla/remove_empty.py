import os
from os import listdir
from os.path import isfile, join
from typing import List

labels_path = os.path.abspath("labels")
label_files: List[str] = [labels_path + "/" + f for f in listdir(labels_path) if isfile(join(labels_path, f))]

for label_file in label_files:
    if os.path.getsize(label_file) == 0:
        os.remove(label_file)
        os.remove(label_file.replace("labels", "images").replace("txt", "jpg"))
