import os
import csv

# path = (
#    "rp_data/train/stroke"  # Replace with the path to the directory you want to search
# )
#
# count = 0
#
# for filename in os.listdir(path):
#    if "Store" not in filename:
#        count += 1
#
# print(f"The number of files is {count}.")

with open("rp_data\\stroke.csv", mode="r", newline="") as csv_file:
    reader = csv.reader(csv_file)

    num_rows = len(list(reader))

print(num_rows)
