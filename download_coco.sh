#!/bin/bash

# Specify download directory
download_dir="data/coco"

# Create download directory if it doesn't exist
mkdir -p "$download_dir"

# Download Images
echo "Downloading COCO 2017 Images..."
wget -c http://images.cocodataset.org/zips/train2017.zip -P "$download_dir"
wget -c http://images.cocodataset.org/zips/val2017.zip -P "$download_dir"
wget -c http://images.cocodataset.org/zips/test2017.zip -P "$download_dir"

# Download Annotations
echo "Downloading COCO 2017 Annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P "$download_dir"

# Unzip Files
echo "Extracting Images..."
unzip -q "$download_dir/train2017.zip" -d "$download_dir"
unzip -q "$download_dir/val2017.zip" -d "$download_dir"
unzip -q "$download_dir/test2017.zip" -d "$download_dir"

echo "Extracting Annotations..."
unzip -q "$download_dir/annotations_trainval2017.zip" -d "$download_dir"

# Clean up zip files
rm $download_dir/*.zip

echo "COCO 2017 dataset download and extraction complete!"