import os
import shutil

# Define your source and destination directories (use raw strings)
source_dir = r'C:\github\resnet-pytorch\dataset\val\PNEUMONIA'
bacteria_dir = r'C:\github\resnet-pytorch\BACTERIA'
virus_dir = r'C:\github\resnet-pytorch\VIRUS'

# Create destination directories if they don't exist
os.makedirs(bacteria_dir, exist_ok=True)
os.makedirs(virus_dir, exist_ok=True)

# Loop through files and move them
for filename in os.listdir(source_dir):
    src_path = os.path.join(source_dir, filename)

    if os.path.isfile(src_path):
        if 'bacteria' in filename.lower():
            shutil.move(src_path, os.path.join(bacteria_dir, filename))
        elif 'virus' in filename.lower():
            shutil.move(src_path, os.path.join(virus_dir, filename))