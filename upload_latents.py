import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Upload Parquet shards to Hugging Face Hub")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the output directory containing the 'parquet_shards' folder (e.g., ./data/math_qwen_2.5_7B_layer_21_latents)"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True, 
        help="Target Hugging Face repository ID (e.g., KingTechnician/math_qwen_2.5_7B_layer_21_latents)"
    )
    
    args = parser.parse_args()

    # Automatically append 'parquet_shards' to the provided directory
    parquet_dir = os.path.join(args.data_dir, "parquet_shards")

    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Could not find the parquet shards directory at: {parquet_dir}")

    print(f"Loading Parquet shards from: {parquet_dir}")

    # Load all parquet shards natively into a Hugging Face Dataset
    try:
        ds = load_dataset("parquet", data_dir=parquet_dir, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Successfully loaded dataset with {len(ds)} rows.")
    print(f"Columns: {ds.column_names}")

    # Push to Hugging Face Hub
    print(f"Pushing to Hugging Face Hub: {args.repo_id}...")
    try:
        ds.push_to_hub(args.repo_id)
        print(f"✅ Upload complete! Successfully pushed to {args.repo_id}")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        print("Tip: Make sure you have run `huggingface-cli login` first.")

if __name__ == "__main__":
    main()
