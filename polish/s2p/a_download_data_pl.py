from datasets import load_dataset
import os

# 1. Define your custom cache directory on the /scratch partition
#    It's good practice to create a specific folder for it.
cache_path = "/scratch/ABHI/huggingface_cache2"

# 2. Make sure this directory exists
os.makedirs(cache_path, exist_ok=True)

print(f"--- Starting dataset download ---")
print(f"Saving to cache: {cache_path}")

# 3. Call load_dataset and pass your custom path to the 'cache_dir' argument
cv_22 = load_dataset(
    "fsicoli/common_voice_22_0", 
    "pl", 
    split="train", 
    trust_remote_code=True,
    cache_dir=cache_path  # This is the line you need to add
)

print("--- Dataset downloaded and loaded successfully! ---")
print(cv_22)