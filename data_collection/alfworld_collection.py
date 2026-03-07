#!/usr/bin/env python3
"""
ALFWorld Data Collection with Command Line Arguments

This script collects ALFWorld data using a language model, with all parameters
configurable via command line arguments instead of environment variables.

Usage:
    python alfworld_collection_args.py --dataset_path ./alfworld_dataset.json --output_dir ./output
"""
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import glob
import json
import pickle
import time
import argparse
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import sys
import datasets
from datasets import Features, Value, Sequence, load_from_disk
from datasets import Dataset as HFDataset
import pandas as pd
import re
import socket
import datetime
import math
from pathlib import Path

# Global variable to collect all data
all_data = []

def save_train_data(task, task_id, plan, hidden_state):
    """Collect task, task_id, and hidden_state for each sample into a global list"""
    
    # Safely handle both batched [1, T, H] and sliced [T, H] inputs
    if hidden_state.dim() == 3 and hidden_state.size(0) == 1:
        hidden_np = hidden_state.to(torch.float32).cpu().numpy().squeeze(0).astype(np.float32)  # [T, H]
    else:
        hidden_np = hidden_state.to(torch.float32).cpu().numpy().astype(np.float32)             # [T, H]

    entry = {
        "task": task,
        "task_id": task_id,
        "plan": plan,
        "hidden_state": hidden_np
    }

    all_data.append(entry)
    return None


def convert_to_hf_dataset(data, output_dir="final_output"):
    """Convert collected data to a HuggingFace Dataset and save as Dataset and Parquet files"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert data format
    tasks = [d['task'] for d in data]
    task_ids = [d['task_id'] for d in data]
    plan = [d['plan'] for d in data]
    hidden_states = [d['hidden_state'].astype(np.float32) for d in data]  # list of ndarrays [T, D]

    # Define feature schema
    features = Features({
        'task': Value('string'),
        'task_id': Value('string'),
        'plan': Value('string'),
        'hidden_state': Sequence(Sequence(Value('float32')))  # support variable length [T, D]
    })

    # Create dataset
    hf_dataset = HFDataset.from_dict({
        "task": tasks,
        "task_id": task_ids,
        "plan": plan,
        "hidden_state": [hs.tolist() for hs in hidden_states]  # ndarray -> list
    }, features=features)

    # Save to disk (HuggingFace Dataset format)
    hf_dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    print(f"✅ HuggingFace Dataset saved to {os.path.join(output_dir, 'hf_dataset')}")

    # Also save as Parquet
    parquet_path = os.path.join(output_dir, "data.parquet")
    hf_dataset.to_parquet(parquet_path)
    print(f"✅ Parquet file saved to {parquet_path}")

def save_rank_data(rank, all_data, temp_dir="temp_rank_data"):
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"rank_{rank}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Rank {rank} saved data to {filename}")

def finalize_data_save_and_merge(rank, world_size, output_dir="final_output", temp_dir="temp_rank_data"):
    import glob
    
    chunk_dir = os.path.join(output_dir, "safe_chunks")
    all_chunks = glob.glob(os.path.join(chunk_dir, "*.pkl"))
    
    if not all_chunks:
        print("No chunks found. Nothing to save.")
        return

    print(f"Found {len(all_chunks)} safe chunks. Converting directly to Parquet shards...")
    
    shards_dir = os.path.join(output_dir, "parquet_shards")
    os.makedirs(shards_dir, exist_ok=True)
    
    for i, chunk_file in enumerate(all_chunks):
        try:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
            
            out_path = os.path.join(shards_dir, f"data_shard_{i:05d}.parquet")
            _write_parquet_shard(chunk_data, out_path)
            print(f"✅ Converted {os.path.basename(chunk_file)} -> {os.path.basename(out_path)}")
            
            del chunk_data
            gc.collect()
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")

    print(f"\n🎉 ALFWorld data safely stored in {shards_dir}!")


def setup_distributed(args):
    """Initialize distributed training environment"""
    # Print current environment for debugging
    print("Environment variables before setup:")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"Hostname: {socket.gethostname()}")

    # Get system-wide GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Available GPU count: {gpu_count}")

    # Respect existing environment variables
    if 'LOCAL_RANK' not in os.environ and args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Get rank and world_size
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size or gpu_count))

    print(f"Using rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    try:
        # Set device before initializing process group
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=datetime.timedelta(minutes=args.distributed_timeout),
            init_method=args.init_method,
            world_size=world_size,
            rank=rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process group initialized. Rank: {rank}, World Size: {world_size}")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to single-process mode
        rank = 0
        local_rank = 0
        world_size = 1
        print("Falling back to single-process mode")

    return rank, world_size


class MMDataset(Dataset):
    """Dataset class for MMLU data"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_model(model_path, rank, torch_dtype="float32"):
    """Load model and prepare for DDP"""
    device = torch.device(f"cuda:{rank}")

    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_obj = dtype_mapping.get(torch_dtype, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_obj,
        device_map={"": device}
    )

    print(f"Model initialized on GPU {rank}.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def agent_generate(model, tokenizer, texts, device, args):
    """Modified to handle a list of texts (batch) and specific layer indexing"""
    formatted_texts = [
        '<|im_start|>user\n' + t + '<|im_end|>\n<|im_start|>assistant\n'
        for t in texts
    ]
    
    tokenizer.padding_side = 'left' 
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_length = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

    generated_texts = tokenizer.batch_decode(
        outputs.sequences[:, input_length:], skip_special_tokens=True
    )

    # Correct extraction: target layer, last position at each step
    step_hiddens = []
    steps = outputs.hidden_states
    start_index = max(0, len(steps) - args.max_hidden_states)
    for i in range(start_index, len(steps)):
        target_layer = steps[i][args.layer_index]  # [B, S_i, H] <- Uses dynamic layer index
        h_last = target_layer[:, -1, :]            # [B, H]
        step_hiddens.append(h_last)

    hidden_seq = torch.stack(step_hiddens, dim=1) # [B, T, H]

    return generated_texts, hidden_seq


def infer_chain(model, tokenizer, tasks, task_ids, device, args):
    """Modified to process batched arrays"""
    ALFWORLD_TEMPLATE = '''
    Please provide a general plan to solve this task.\\n

    The task is: {task}
    '''
    prompts = [ALFWORLD_TEMPLATE.format(task=t) for t in tasks]
    
    generated_texts, hidden_seqs = agent_generate(
        model, tokenizer, prompts, device, args
    )

    batch_size = len(tasks)
    for i in range(batch_size):
        plan = generated_texts[i]
        if args.verbose:
            print(f"Agent output for {task_ids[i]}: {plan[:100]}...")

        # hidden_seqs[i] slice automatically matches the [T, H] shape expected by save_train_data
        save_train_data(task=tasks[i], task_id=task_ids[i], plan=plan, hidden_state=hidden_seqs[i])

    return hidden_seqs

def _write_parquet_shard(rows: list, out_path: str):
    """Write rows directly to a single Parquet file to save memory"""
    hs_type = pa.list_(pa.list_(pa.float32()))
    schema = pa.schema([
        pa.field('task', pa.string()),
        pa.field('task_id', pa.string()),
        pa.field('plan', pa.string()),
        pa.field('hidden_state', hs_type),
    ])

    tasks        = [r['task'] for r in rows]
    task_ids     = [r['task_id'] for r in rows]
    plans        = [r['plan'] for r in rows]
    
    # Ensure nested list format for hidden states
    hidden_lists = [
        (r['hidden_state'].tolist() if isinstance(r['hidden_state'], np.ndarray) else r['hidden_state'])
        for r in rows
    ]

    table = pa.table({
        'task': pa.array(tasks, type=pa.string()),
        'task_id': pa.array(task_ids, type=pa.string()),
        'plan': pa.array(plans, type=pa.string()),
        'hidden_state': pa.array(hidden_lists, type=hs_type),
    }, schema=schema)

    pq.write_table(table, out_path, compression="zstd", use_dictionary=True)



def evaluate(model, tokenizer, dataloader, device, rank, world_size, args):
    global all_data
    task_num = 0
    correct_count = 0
    import glob

    # Create a directory for safe incremental backups
    chunk_dir = os.path.join(args.output_dir, "safe_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # --- RESUME LOGIC: Find the highest batch index saved ---
    start_batch_idx = 0
    existing_chunks = glob.glob(os.path.join(chunk_dir, "chunk_*.pkl"))
    if existing_chunks:
        # Extract batch numbers from filenames like "chunk_5.pkl"
        saved_indices = []
        for f in existing_chunks:
            basename = os.path.basename(f)
            if "final" not in basename:
                try:
                    num = int(basename.split('_')[1].split('.pkl')[0])
                    saved_indices.append(num)
                except ValueError:
                    pass
        if saved_indices:
            start_batch_idx = max(saved_indices) + 1
            print(f"\n🔄 Resuming from batch index {start_batch_idx} (found {len(existing_chunks)} saved chunks).")

    pbar = tqdm(total=len(dataloader), desc=f"GPU {rank} processing")
    
    # Update progress bar to reflect already completed batches
    if start_batch_idx > 0:
        pbar.update(start_batch_idx)

    try:
        for batch_idx, task_item in enumerate(dataloader):
            # --- RESUME LOGIC: Skip already processed batches ---
            if batch_idx < start_batch_idx:
                task_num += len(task_item['tasks'])
                continue

            tasks = task_item['tasks']
            task_ids = task_item['task_ids']
            
            current_batch_size = len(tasks)
            task_num += current_batch_size

            if args.verbose:
                print(f"GPU {rank}, batch {batch_idx}/{len(dataloader)}, processing {current_batch_size} items...")

            _ = infer_chain(model, tokenizer, tasks, task_ids, device, args)
            
            # --- INCREMENTAL SAVING TO PREVENT OOM ---
            if len(all_data) >= 64:  
                chunk_path = os.path.join(chunk_dir, f"chunk_{batch_idx}.pkl")
                with open(chunk_path, "wb") as f:
                    pickle.dump(all_data, f)
                print(f"\n💾 [Memory Saver] Flushed {len(all_data)} items to disk. Clearing RAM...")
                all_data.clear()
                gc.collect()

            pbar.update(1)
    except Exception as e:
        print(f"GPU {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()

    # Save any remaining data in the buffer at the very end
    if len(all_data) > 0:
        chunk_path = os.path.join(chunk_dir, f"chunk_final.pkl")
        with open(chunk_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"\n💾 [Memory Saver] Flushed final {len(all_data)} items to disk.")
        all_data.clear()
        gc.collect()

    return correct_count, task_num




def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="ALFWorld Data Collection with Configurable Parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path", type=str,
        default="Interlat_preview/models/Qwen2.5-7B",
        help="Path to the model (HuggingFace model name or local path)"
    )
    model_group.add_argument(
        "--torch_dtype", type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="PyTorch dtype for the model"
    )

    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        "--dataset_path", type=str,
        default="Interlat_preview/datasets/alfworld_sft.json",
        help="Path to the ALFWorld dataset JSON file"
    )
    data_group.add_argument(
        "--output_dir", type=str,
        help="Output directory for collected data (default: auto-generated from temperature)"
    )

    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for text generation"
    )
    gen_group.add_argument(
        "--max_new_tokens", type=int, default=5000,
        help="Maximum number of new tokens to generate"
    )
    gen_group.add_argument(
        "--do_sample", action="store_true", default=True,
        help="Whether to use sampling for generation"
    )
    gen_group.add_argument(
        "--no_sample", dest="do_sample", action="store_false",
        help="Disable sampling (use greedy decoding)"
    )
    gen_group.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    gen_group.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling parameter"
    )
    gen_group.add_argument(
        "--num_beams", type=int, default=1,
        help="Number of beams for beam search"
    )
    gen_group.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="Repetition penalty for generation"
    )
    gen_group.add_argument(
        "--max_hidden_states", type=int, default=10000,
        help="Maximum number of hidden states to collect"
    )
    gen_group.add_argument(
        "--layer_index", type=int, default=-1,
        help="The specific hidden layer to extract (-1 for the last layer)"
    )


    # Distributed training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    dist_group.add_argument(
        "--world_size", type=int, default=None,
        help="World size for distributed training (auto-detected if not specified)"
    )
    dist_group.add_argument(
        "--distributed_backend", type=str, default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend"
    )
    dist_group.add_argument(
        "--distributed_timeout", type=int, default=300,
        help="Distributed training timeout in minutes"
    )
    dist_group.add_argument(
        "--init_method", type=str, default="env://",
        help="Initialization method for distributed training"
    )

    # Storage configuration
    storage_group = parser.add_argument_group('Storage Configuration')
    storage_group.add_argument(
        "--temp_dir", type=str, default="temp_rank_data",
        help="Temporary directory for rank data during distributed processing"
    )

    # Miscellaneous
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging"
    )
    misc_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for data processing"
    )
    misc_group.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of workers for data loading"
    )

    return parser


def validate_arguments(args):
    """Validate and process arguments"""
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./alfworld_data_temp_{args.temperature}"

    # Validate paths
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate model path (if local)
    if not args.model_path.startswith(('http://', 'https://')) and not '/' in args.model_path:
        # Assume it's a HuggingFace model name, no validation needed
        pass
    elif os.path.exists(args.model_path):
        # Local model path exists
        pass
    elif '/' in args.model_path and not os.path.exists(args.model_path):
        # Might be HuggingFace format like "Qwen/Qwen2.5-7B-Instruct"
        print(f"Model path appears to be HuggingFace format: {args.model_path}")

    return args

def custom_collate_fn(batch):
    """
    Extract only the necessary string fields to avoid PyTorch's default_collate
    crashing on uneven list lengths in the 'conversations' field.
    """
    tasks = []
    task_ids = []
    
    for item in batch:
        # We know from the dataset structure that the actual instruction 
        # is always in conversations[2]['value']
        tasks.append(item['conversations'][2]['value'])
        task_ids.append(item['id'])
        
    return {
        'tasks': tasks,
        'task_ids': task_ids
    }


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_arguments(args)

    # Print configuration
    print("=" * 50)
    print("ALFWorld Data Collection Configuration")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Torch dtype: {args.torch_dtype}")
    print("=" * 50)

    # Initialize distributed environment
    rank, world_size = setup_distributed(args)
    print(f"After setup: Rank = {rank}, World Size = {world_size}")

    # Device setup
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f"cuda:{local_rank}")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, local_rank, args.torch_dtype)

    if rank == 0:
        print(f'Dataset path: {args.dataset_path}')
        print(f'Output dir: {args.output_dir}')

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        full_train_dataset = json.load(f)

    print(f"Training set size: {len(full_train_dataset)}")

    full_train_dataset = HFDataset.from_list(full_train_dataset)

    # Data sharding
    if world_size > 1:
        total_samples = len(full_train_dataset)
        samples_per_worker = math.ceil(total_samples / world_size)
        start_idx = rank * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        my_indices = list(range(start_idx, end_idx))
        my_dataset = full_train_dataset.select(my_indices)
        print(
            f"GPU {rank} handling samples {start_idx} to {end_idx-1}, "
            f"total: {len(my_dataset)}"
        )
    else:
        my_dataset = full_train_dataset
        print(f"Single process mode - handling all {len(my_dataset)} samples")

    mm_dataset = MMDataset(my_dataset)
    dataloader = DataLoader(
        mm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    print(f"GPU {rank} number of samples to process: {len(dataloader)}")

    # Start evaluation
    correct_count, task_num = evaluate(
        model, tokenizer, dataloader, device, rank, world_size, args
    )

    # Synchronize processes
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            print(f"Barrier error on GPU {rank}: {e}")

    # Final data save
    finalize_data_save_and_merge(
        rank, world_size, output_dir=args.output_dir, temp_dir=args.temp_dir
    )

    # Cleanup
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error destroying process group on GPU {rank}: {e}")

    print(f"GPU {rank} completed processing {task_num} tasks.")


if __name__ == "__main__":
    main()