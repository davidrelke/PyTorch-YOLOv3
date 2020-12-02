import math
import os
from os import listdir
from os.path import isfile, join
import random
from typing import List

images_path = os.path.abspath("images")

image_files: List[str] = [images_path + "/" + f for f in listdir(images_path) if isfile(join(images_path, f))]

with open("train.txt", "w") as file_train:
    with open("val.txt", "w") as file_val:
        num_train = math.ceil(len(image_files) * 0.75)
        num_val = len(image_files) - num_train
        train_list = random.sample(image_files, num_train)
        count_train = 0
        count_val = 0

        for file in train_list:
            file_train.write(f"{file}\n")
            count_train += 1

        for file in image_files:
            if file not in train_list:
                file_val.write(f"{file}\n")
                count_val += 1

        print(count_train / len(image_files))
        print(count_val / len(image_files))