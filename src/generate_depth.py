# Import dependencies
import os
from os.path import join
from glob import glob
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
import requests

print("Zero-shot image-to-depth generation\n")

# Settings
image_dir = join(os.pardir, 'data', 'coco', 'train2017')
depth_dir = join(os.pardir, 'data', 'coco', 'depth', 'train2017')

print(f"Image dir: {image_dir}")
print(f"Depth dir: {depth_dir}")

if not os.path.isdir(depth_dir):
    os.mkdir(depth_dir)

assert os.path.isdir(image_dir), f"Directory not found: [{image_dir}]"
assert os.path.isdir(depth_dir), f"Directory not found: [{depth_dir}]"

# Get the paths of available images
image_paths = glob(join(image_dir, '*.jpg'))
print(f"\nFound {len(image_paths)} images...")

# Load depth-anything
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# Generate depth images
print("\nGenerating depth images...\n")
for i, path in tqdm(enumerate(image_paths), total=len(image_paths)):
    image = Image.open(path)
    depth = pipe(image)["depth"]
    depth_name = image_paths[i].split('/')[-1].split('.')[0] + "_depth.jpg"
    depth.save(join(depth_dir, depth_name))
print("\nDone\n\n")
