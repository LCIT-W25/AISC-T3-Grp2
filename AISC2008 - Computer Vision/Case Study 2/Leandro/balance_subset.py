import os
import random
import shutil
import pandas as pd
import json

# Set paths
source_folder = "data/photos"  
destination_folder = "data/balanced_photos"  
metadata_file = "data/photos.json" 
output_json = "data/balanced_subset.json"  

# Desired total number of images and per-label images
desired_total_images = 100000
labels_count = 5  # Number of unique labels (adjust as needed)
images_per_label = desired_total_images // labels_count

# Load metadata (assumes a CSV file with 'photo_id' and 'label' columns)
metadata = pd.read_json(metadata_file, lines=True)

# Check if columns exist
if 'photo_id' not in metadata.columns or 'label' not in metadata.columns:
    raise ValueError("The metadata file must contain 'photo_id' and 'label' columns.")

# Create a balanced subset
balanced_subset = []
unique_labels = metadata['label'].unique()

for label in unique_labels:
    label_subset = metadata[metadata['label'] == label]
    sampled_photos = label_subset.sample(min(len(label_subset), images_per_label)).to_dict(orient="records")
    balanced_subset.extend(sampled_photos)

# Shuffle the subset
random.shuffle(balanced_subset)

# Save the balanced subset to a JSON file
with open(output_json, "w") as f:
    json.dump(balanced_subset, f, indent=4)

print(f"Balanced subset with {len(balanced_subset)} images saved to {output_json}.")

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy the selected images to the destination folder
for photo in balanced_subset:
    photo_id = photo['photo_id']
    source_path = os.path.join(source_folder, f"{photo_id}.jpg")
    dest_path = os.path.join(destination_folder, f"{photo_id}.jpg")
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
    else:
        print(f"Warning: File {photo_id}.jpg not found in source folder.")

print(f"Copied {len(balanced_subset)} images to {destination_folder}.")