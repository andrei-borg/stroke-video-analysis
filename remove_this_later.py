import os

path = "rp_data/valid/normal" # Replace with the path to the directory you want to search

count = 0

for filename in os.listdir(path):
    if 'normal' in filename:
        count += 1

print(f'The number of files with "normal" in their name is {count}.')
