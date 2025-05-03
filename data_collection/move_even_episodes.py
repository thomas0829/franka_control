import sys
import os
import shutil
import re

source = sys.argv[1]
destination = sys.argv[2]

os.makedirs(destination, exist_ok=True)
pattern = re.compile(r"^episode_(\d+)\.npy$")
files = os.listdir(source)

for filename in files:
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        if number % 2 == 0:
            source_path = os.path.join(source, filename)
            destination_path = os.path.join(destination, filename)
            if os.path.isfile(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved: {filename}")