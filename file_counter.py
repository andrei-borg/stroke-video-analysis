import os
import csv

path = "D:\\Kandidatarbete\\rp_eyes\\train\\normal"  # Replace with the path to the directory you want to search
count = 0
for filename in os.listdir(path):
    if "Store" not in filename:
        count += 1
print(f"The number of files is {count}.")