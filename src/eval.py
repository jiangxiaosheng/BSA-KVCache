from config import SimConfig, sim_config
import os
from tqdm import tqdm
import numpy as np
from simulator import setup_cache
import logging
import re
from collections import defaultdict
from kvcache import BsaKVCache
import argparse
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def validate_traces(trace_dir: str, config: SimConfig):
    all_files = set(os.listdir(trace_dir))

    # Extract unique request numbers
    trace_pattern = re.compile(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy")
    my_seq_id = -1
    trace_files = defaultdict(lambda: defaultdict(dict))

    for filename in all_files:
        match = trace_pattern.match(filename)
        if match:
            seq_id = int(match.group(1))
            if my_seq_id == -1:
                my_seq_id = seq_id
            elif my_seq_id != seq_id:
                raise ValueError(f"Sequence Id mismatch: {my_seq_id} != {seq_id}")

            layer_idx = int(match.group(2))
            trace_type = match.group(3)
            assert trace_type in [
                "blocks",
                "scores",
            ], "Unknown trace type: {trace_type}. Should be blocks or scores"

            trace_files[layer_idx][trace_type] = os.path.join(
                trace_dir, filename
            )

    if not trace_files:
        raise ValueError(f"No trace files found in {trace_dir}")

    for layer_idx in range(config.num_layers):
        if layer_idx not in trace_files:
            raise ValueError(f"Layer {layer_idx} not found for sequence {my_seq_id}")
        layer_data = trace_files[layer_idx]
        if "blocks" not in layer_data:
            raise ValueError(
                f"Blocks not found for sequence {my_seq_id} and layer {layer_idx}"
            )
        if "scores" not in layer_data:
            raise ValueError(
                f"Scores not found for sequence {my_seq_id} and layer {layer_idx}"
            )

    return trace_files


def preload_traces(trace_files: dict, config: SimConfig):
    traces = defaultdict(dict)
    for i in range(config.num_layers):
        block_trace = np.load(trace_files[i]["blocks"])
        score_trace = np.load(trace_files[i]["scores"])
        traces[i] = (block_trace, score_trace)
    return traces


def get_num_traces(config: SimConfig) -> int:
    return len(os.listdir(config.trace_dir))


def eval_avg_mem_usage(config_path: str, out_path: str):
    config = sim_config()
    config.from_yaml(config_path)
    print(f"Config: {config}")
    num_traces = get_num_traces(config)
    print(f"Number of traces: {num_traces}")


    num_traces = 10
    trace_idx_to_spoton = 0
    mem_usages_bsa = []
    mem_usages_paged = []
    mem_usages_bsa_spoton = []
    mem_usage_paged_spoton = 0
    for i in tqdm(range(num_traces), desc="Evaluating traces"):
        trace_dir = os.path.join(config.trace_dir, f"trace{i:04d}")
        trace_files = validate_traces(trace_dir, config)
        traces = preload_traces(trace_files, config)
        
        num_iters = traces[0][0].shape[2]

        mem_usage = 0
        for cur_iter in range(num_iters):
            mem_usage_per_iter = 0
            for cur_layer in range(config.num_layers):
                block_ids = traces[cur_layer][0][:, :, cur_iter]

                # follow GQA pattern, i.e. one single KV head can be shared across multiple Q heads
                for kvhead_id in range(
                    config.num_heads // config.kv_group_size
                ):
                    start_col = kvhead_id * config.kv_group_size
                    end_col = (kvhead_id + 1) * config.kv_group_size
                    cur_kvhead_block_ids = block_ids[:, start_col:end_col]

                    cur_kvhead_block_ids = cur_kvhead_block_ids.flatten().tolist()
                    cur_kvhead_block_ids = set(cur_kvhead_block_ids)
                    mem_usage_per_iter += len(cur_kvhead_block_ids) * BsaKVCache.bytes() / 1024 / 1024 / 1024
            mem_usage += mem_usage_per_iter

            if i == trace_idx_to_spoton:
                mem_usages_bsa_spoton.append(mem_usage_per_iter)
        
        paged_mem_usage = 4096 * config.num_layers * config.num_kvheads * config.hidden_size * 2 * config.dtype_bytes / 1024 / 1024 / 1024
        if i == trace_idx_to_spoton:
            mem_usage_paged_spoton = paged_mem_usage

        mem_usages_bsa.append(mem_usage / num_iters)
        mem_usages_paged.append(paged_mem_usage)

    # print(f"Mem usage BSA: {mem_usages_bsa}")
    # print(f"Mem usage Paged: {mem_usages_paged}")
    # print(f"Mem usage BSA Spoton: {mem_usages_bsa_spoton}")
    # print(f"Mem usage Paged Spoton: {mem_usage_paged_spoton}")
    
    # Plot 1: Memory usage over iterations with paged baseline
    plt.figure(figsize=(10, 6))
    iterations = np.arange(len(mem_usages_bsa_spoton))
    plt.plot(iterations, mem_usages_bsa_spoton, marker='o', linewidth=2, markersize=4, label='BSA KV Cache Memory Usage')
    plt.axhline(y=mem_usage_paged_spoton, color='r', linestyle='--', linewidth=2, label='Paged KV Cache Memory Usage')
    plt.xlabel('Decode Step', fontsize=12)
    plt.ylabel('Memory Usage (GB)', fontsize=12)
    # plt.title('KV Cache Memory Usage Over Iterations', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = f"{out_path}_single_sequence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 1 saved to {save_path}")
    
    # Plot 2: CDF of relative memory usage (BSA / Paged)
    relative_mem_usages = np.array(mem_usages_bsa) / np.array(mem_usages_paged)
    sorted_relative = np.sort(relative_mem_usages)
    cdf = np.arange(1, len(sorted_relative) + 1) / len(sorted_relative)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_relative, cdf, linewidth=2)
    plt.xlabel('Relative Memory Usage (BSA KV Cache / Paged KV Cache)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    # plt.title('CDF of Relative Memory Usage Across Traces', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save CDF plot with modified filename
    base_path = out_path.rsplit('.', 1)[0]
    cdf_path = f"{base_path}_cdf.png"
    plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
    print(f"Plot 2 (CDF) saved to {cdf_path}")
    
    # print(f"Average relative memory usage: {np.mean(relative_mem_usages):.4f}")
    # print(f"Median relative memory usage: {np.median(relative_mem_usages):.4f}")
    # print(f"Min relative memory usage: {np.min(relative_mem_usages):.4f}")
    # print(f"Max relative memory usage: {np.max(relative_mem_usages):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval_avg_mem_usage(args.config, args.out)