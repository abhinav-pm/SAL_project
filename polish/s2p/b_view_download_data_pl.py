from datasets import load_dataset, Audio
import pandas as pd

# Define the same cache directory
cache_path = "/scratch/ABHI/huggingface_cache"

print("--- Loading dataset from cache ---")

# Load the dataset with audio decoding disabled
cv_22 = load_dataset(
    "fsicoli/common_voice_22_0", 
    "de", 
    split="train", 
    cache_dir=cache_path
)

# Disable audio decoding to avoid the FFmpeg error
cv_22 = cv_22.cast_column("audio", Audio(decode=False))

print("--- Dataset loaded successfully! ---")
print(cv_22)

# ===== EXPLORING THE DATASET =====

# 1. See basic information
print("\n--- Dataset Info ---")
print(f"Number of examples: {len(cv_22)}")
print(f"Column names: {cv_22.column_names}")

# 2. Look at first example (now it won't try to decode audio)
print("\n--- First Example ---")
print(cv_22[0])

# 3. Look at first few examples as DataFrame
print("\n--- First 10 Examples (as DataFrame) ---")
df = pd.DataFrame(cv_22[:10])
print(df)

# 4. See sample sentences
print("\n--- Sample Sentences ---")
for i in range(min(5, len(cv_22))):
    print(f"{i+1}. {cv_22[i]['sentence']}")

# 5. Basic statistics
print("\n--- Basic Statistics ---")
print(f"Total examples: {len(cv_22)}")

# Convert to full DataFrame for statistics (this might take a moment)
print("\nGenerating statistics...")
df_full = pd.DataFrame(cv_22[:])
print(f"\nGender distribution:")
print(df_full['gender'].value_counts())
print(f"\nAge distribution:")
print(df_full['age'].value_counts())
print(f"\nAccent distribution:")
print(df_full['accent'].value_counts())