import os

images_path = os.path.abspath("images") + "\\"

with open("train.txt", "r") as file_train:
    lines_train = file_train.readlines()

with open("val.txt", "r") as file_val:
    lines_val = file_val.readlines()

with open("train.txt", "w") as file_train:
    for line in lines_train:
        file_train.write(line.replace("data/images/", images_path))

with open("val.txt", "w") as file_val:
    for line in lines_val:
        file_val.write(line.replace("data/images/", images_path))
