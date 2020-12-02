import os
from os import listdir
from os.path import isfile, join
from typing import List

from PIL import Image

labels_path = os.path.abspath("labels")
label_files: List[str] = [labels_path + "/" + f for f in listdir(labels_path) if isfile(join(labels_path, f))]

num_cars = 0
num_trucks = 0
num_bikes = 0
num_motorbikes = 0
num_person = 0
for file in label_files:
    with open(file, "r") as label_file:
        for line in label_file.readlines():
            if line.startswith("0"):
                num_cars = num_cars + 1
            elif line.startswith("1"):
                num_trucks = num_trucks + 1
            elif line.startswith("2"):
                num_motorbikes = num_motorbikes + 1
                image_obj = Image.open(file.replace("labels", "images").replace(".txt", ".jpg"))
                rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
                rotated_image.save(file.replace("labels", "images").replace(".txt", "_rotated.jpg"))
                rotated_image.show()
            elif line.startswith("3"):
                num_bikes = num_bikes + 1
            elif line.startswith("4"):
                num_person = num_person + 1

print(f"cars: {num_cars} trucks: {num_trucks} bikes: {num_bikes} motorbikes: {num_motorbikes} person: {num_person}")